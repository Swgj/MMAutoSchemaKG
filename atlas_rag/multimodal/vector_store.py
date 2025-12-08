import os
import time
import logging
from tqdm import tqdm
from neo4j import GraphDatabase
from openai import OpenAI
from dotenv import load_dotenv
from atlas_rag.llm_generator.prompt.mmkg_prompt import IMAGE_CAPTION_PROMPT
from atlas_rag.llm_generator import LLMGenerator
from atlas_rag.vectorstore.embedding_model import EmbeddingAPI
from atlas_rag.multimodal.utils import url_to_base64

load_dotenv()
logger = logging.getLogger(__name__)

class MultimodalVectorStore:
    """
    Multimodal Vector Store Manager.
    Handles:
    1. Image Captioning (using VLM like GPT-4o)
    2. Vector Index Creation in Neo4j
    3. Embedding Calculation (using shared EmbeddingAPI)
    """
    def __init__(self, uri, user, password, vlm_model="gemini-2.5-flash", embedding_model="gemini-embedding-001"):
        # 1. Neo4j Driver
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        
        # 2. OpenAI Client & VLM Config
        self.llm_client = OpenAI(api_key=os.getenv("GEMINI_API_KEY"), base_url=os.getenv("GEMINI_BASE_URL"))
        self.vlm_model = vlm_model
        self.vlm_generator = LLMGenerator(
            client=self.llm_client,
            model_name=self.vlm_model,
            backend='openai',
            max_workers=4,
        )
        
        # 3. Embedding Model
        self.emb_model = EmbeddingAPI(
            emb_client=self.llm_client, 
            model_name=embedding_model
        )

    def close(self):
        self.driver.close()

    def create_vector_indexes(self, dim=1536):
        """
        Create vector indexes for all node types in Neo4j.
        Default dimension is 1536.
        """
        indexes = [
            ("episode_index", "Episode"),
            ("entity_index", "Entity"),
            ("event_index", "Event"),
            ("image_index", "Image") # 图片也存文本向量(基于描述)
        ]
        
        with self.driver.session() as session:
            for name, label in indexes:
                try:
                    # Neo4j 5.x+ 语法
                    session.run(f"""
                        CREATE VECTOR INDEX {name} IF NOT EXISTS
                        FOR (n:{label}) ON (n.embedding)
                        OPTIONS {{indexConfig: {{
                            `vector.dimensions`: {dim},
                            `vector.similarity_function`: 'cosine'
                        }}}}
                    """)
                    logger.info(f"Vector Index '{name}' check passed for :{label}")
                except Exception as e:
                    logger.warning(f"Index creation warning for {label} (Check your Neo4j version): {e}")

    def generate_image_captions(self, batch_size=10):
        """
        [Step 1] Generate descriptions for images using VLM.
        Stores the result in the 'description' property of Image nodes.
        """
        logger.info(f"Generating image captions using {self.vlm_model}...")
        
        with self.driver.session() as session:
            # 查找所有有 URL 但没有 description 的图片
            count_query = "MATCH (n:Image) WHERE n.description IS NULL AND n.url IS NOT NULL RETURN count(n) AS count"
            total = session.run(count_query).single()["count"]
            logger.info(f"Found {total} images pending description.")
            
            if total == 0: return

            pbar = tqdm(total=total, desc="Captioning")
            
            while True:
                # 1. Fetch Batch
                records = session.run(f"""
                    MATCH (n:Image) 
                    WHERE n.description IS NULL AND n.url IS NOT NULL
                    RETURN n.id AS id, n.url AS url
                    LIMIT {batch_size}
                """).data()
                
                if not records: break
                
                updates = []
                batch_messages = []
                batch_ids = []

                for rec in records:
                    img_id = rec['id']
                    url = rec['url']
                    
                    # 2. Prepare Image Content (URL or Base64)
                    img_content = url
                    if not url.startswith("data:"):
                            base64_str = url_to_base64(url)
                            if base64_str:
                                img_content = base64_str
                    
                    # 3. Call VLM
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": IMAGE_CAPTION_PROMPT},
                                {"type": "image_url", "image_url": {"url": img_content}}
                            ]
                        }
                    ]
                    batch_messages.append(messages)
                    batch_ids.append(img_id)
                    
                if batch_messages:
                    try:
                        responses = self.vlm_generator.generate_response(
                            batch_messages=batch_messages,
                            max_new_tokens=2048,
                        )
                        if isinstance(responses, dict):
                            responses = [responses]
                        for img_id, desc in zip(batch_ids, responses):
                            if not desc or desc == "[]":
                                desc = "No description available"
                            updates.append({'id': img_id, 'desc': desc})
                    except Exception as e:
                        logger.error(f"Error captioning image {img_id}: {e}")
                        
                # 4. Write Back to Neo4j
                if updates:
                    session.run("""
                        UNWIND $updates AS row
                        MATCH (n:Image {id: row.id})
                        SET n.description = row.desc
                    """, {'updates': updates})
                
                pbar.update(len(records))
            
            pbar.close()

    def process_embeddings(self, label, property_key, query_type='passage', batch_size=100):
        """
        [Step 2] Compute embeddings for any node type using EmbeddingAPI.
        """
        logger.info(f"Computing embeddings for :{label} using property '{property_key}'...")
        
        with self.driver.session() as session:
            # 查找有内容但没有 embedding 的节点
            count_query = f"MATCH (n:{label}) WHERE n.embedding IS NULL AND n.{property_key} IS NOT NULL RETURN count(n) AS count"
            total = session.run(count_query).single()["count"]
            logger.info(f"Found {total} nodes pending embedding.")
            
            if total == 0: return

            pbar = tqdm(total=total, desc=f"Embedding :{label}")
            
            while True:
                # 1. Fetch Batch
                records = session.run(f"""
                    MATCH (n:{label}) 
                    WHERE n.embedding IS NULL AND n.{property_key} IS NOT NULL
                    RETURN n.id AS id, n.{property_key} AS content
                    LIMIT {batch_size}
                """).data()
                
                if not records: break
                
                ids = [r['id'] for r in records]
                contents = [r['content'] for r in records]
                
                try:
                    # 2. Call EmbeddingAPI (Batch Encode)
                    # 原 EmbeddingAPI.encode 返回的是 numpy array
                    embeddings = self.emb_model.encode(contents, query_type=query_type)
                    
                    # 3. Prepare Updates
                    updates = []
                    for id, emb in zip(ids, embeddings):
                        updates.append({'id': id, 'embedding': emb.tolist()})
                    
                    # 4. Write Back
                    session.run(f"""
                        UNWIND $updates AS row
                        MATCH (n:{label} {{id: row.id}})
                        SET n.embedding = row.embedding
                    """, {'updates': updates})
                    
                except Exception as e:
                    logger.error(f"Embedding batch failed: {e}")
                    break 
                
                pbar.update(len(records))
            pbar.close()

    def process_relationship_embeddings(self, batch_size=50):
        """
        [Step 3] Compute embeddings for Relationships (Edges).
        Constructs text: "HeadNode Relation TailNode" -> Embedding -> Store on Edge.
        """
        logger.info("Computing embeddings for Relationships (Edges)...")
        
        # 排除结构性边，只计算语义边
        # Note: INVOLVES (Event->Entity) is semantically related, should be included
        excluded_types = ['HAS_CHUNK', 'NEXT', 'MENTIONS', 'CONTAINS_IMAGE', 'CONTAINS_EVENT', 'HAS_IMAGE']
        filter_clause = f"NOT type(r) IN {excluded_types}"
        
        with self.driver.session() as session:
            # 1. 统计待处理的边
            count_query = f"MATCH ()-[r]->() WHERE r.embedding IS NULL AND {filter_clause} RETURN count(r) AS count"
            total = session.run(count_query).single()["count"]
            logger.info(f"Found {total} relationships pending embedding.")
            
            if total == 0: return

            pbar = tqdm(total=total, desc="Embedding Edges")
            
            while True:
                # 2. Fetch a batch of edges (get head and tail node IDs for building text)
                # 使用 elementId(r) 在 Neo4j 5.x+ 唯一标识边
                fetch_query = f"""
                    MATCH (h)-[r]->(t)
                    WHERE r.embedding IS NULL AND {filter_clause}
                    RETURN elementId(r) AS id, h.id AS head, type(r) AS rel, t.id AS tail
                    LIMIT {batch_size}
                """
                records = session.run(fetch_query).data()
                if not records: break
                
                ids = []
                texts = []
                for rec in records:
                    ids.append(rec['id'])
                    # 构建语义文本: "Caroline UPLOADED IMG_..." -> "Caroline uploaded IMG_..."
                    # 将下划线转为空格，全大写转小写，更符合自然语言
                    rel_text = rec['rel'].replace("_", " ").lower()
                    text = f"{rec['head']} {rel_text} {rec['tail']}"
                    texts.append(text)
                
                try:
                    # 3. 计算向量 (query_type='edge')
                    embeddings = self.emb_model.encode(texts, query_type='edge')
                    
                    updates = []
                    for eid, emb in zip(ids, embeddings):
                        updates.append({'id': eid, 'embedding': emb.tolist()})
                    
                    # 4. 批量写回
                    session.run("""
                        UNWIND $updates AS row
                        MATCH ()-[r]->() 
                        WHERE elementId(r) = row.id
                        SET r.embedding = row.embedding
                    """, {'updates': updates})
                    
                except Exception as e:
                    logger.error(f"Edge embedding batch failed: {e}")
                    break
                
                pbar.update(len(records))
            pbar.close()

    def run_all(self):
        """Orchestrate the full pipeline"""
        # 1. Setup Indexes
        self.create_vector_indexes()
        
        # 2. Generate Captions for Images (Prioritize this as it enables Image Embedding)
        self.generate_image_captions()
        
        # 3. Compute Embeddings for all types
        # Episode: Use transcript
        self.process_embeddings("Episode", "transcript", query_type="passage")
        
        # Entity & Event: Use id (name/text)
        self.process_embeddings("Entity", "id", query_type="entity")
        self.process_embeddings("Event", "id", query_type="entity")
        
        # Image: Use description (generated in step 2)
        # Treat description as a "passage" or "entity" depending on length. "passage" is safer.
        self.process_embeddings("Image", "description", query_type="passage")
        
        # Edge embedding
        self.process_relationship_embeddings()
        
        logger.info("All multimodal vector store tasks completed!")

if __name__ == "__main__":
    # 配置区
    URI = "bolt://localhost:7687"
    USER = "neo4j"
    PASSWORD = "password"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    # filter httpx info
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logger.info("Starting Multimodal Vector Store")
    
    store = MultimodalVectorStore(URI, USER, PASSWORD)
    store.run_all()
    store.close()