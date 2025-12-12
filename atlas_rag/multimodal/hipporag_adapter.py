import networkx as nx
import numpy as np
from neo4j import GraphDatabase
from tqdm import tqdm
from typing import Dict, Any, List
import faiss
import logging

from atlas_rag.vectorstore.embedding_model import EmbeddingAPI # for edge embedding

logger = logging.getLogger(__name__)

class Neo4jToHippoAdapter:
    """
    Adapter to load Knowledge Graph from Neo4j into the format required by HippoRAGRetriever.
    1. Loads Graph structure (NetworkX)
    2. Loads pre-computed Node Embeddings from Neo4j
    3. Computes Edge Embeddings on-the-fly (or loads if you stored them)
    4. Build faiss index for retrieval
    """
    def __init__(self, uri, user, password, embedding_model: EmbeddingAPI, database_name: str = "neo4j"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.embedding_model = embedding_model
        self.database_name = database_name

    def close(self):
        self.driver.close()

    def load_data(self) -> Dict[str, Any]:
        """
        Main method to return the 'data' dictionary expected by HippoRAG.
        """
        logger.info("[Adapter] Loading Graph & Embeddings from Neo4j...")
        
        G = nx.DiGraph()
        text_dict = {}
        node_list = []      # List of node IDs
        node_embeddings_list = [] # List of numpy arrays
        text_embeddings_list = [] # List of numpy arrays (should be a subset of node_embeddings_list, for it only contains the embeddings of the Episodes)
        
        edge_list = []      # List of (u, v) tuples
        edge_embeddings_list = [] # List of numpy arrays
        image_map = {}
        
        VECTOR_DIM = 1536

        with self.driver.session(database=self.database_name) as session:
            # --- 1. Load Episodes (Passages) ---
            logger.info("[Adapter] Loading Episodes (Passages)...")
            # file_id is the same as chunk_id
            result = session.run("MATCH (ep:Episode) RETURN ep.id AS id, ep.transcript AS text, ep.embedding AS emb")
            for record in result:
                ep_id = record["id"]
                
                # Build NetworkX node
                G.add_node(ep_id, type="passage", id=ep_id, file_id=ep_id)
                text_dict[ep_id] = record["text"]
                
                # Collect list for indexing
                node_list.append(ep_id)
                
                # Read the stored vector, if not, use zero vector to fill (prevent error)
                emb = record["emb"]
                if emb:
                    node_embeddings_list.append(np.array(emb))
                    text_embeddings_list.append(np.array(emb))
                    VECTOR_DIM = len(emb) # automatically set the vector dimension
                else:
                    # If really no embedding, maybe need to temporary calculate or report error, here give a warning
                    logger.warning(f"Episode {ep_id} has no embedding!")
                    node_embeddings_list.append(np.zeros(VECTOR_DIM))
                    text_embeddings_list.append(np.zeros(VECTOR_DIM))

            # --- 2. Load Entities/Images/Events ---
            logger.info("[Adapter] Loading Entities & Provenance...")
            # Find all nodes connected to Episode via semantic relations, and get provenance (file_id)
            query = """
                MATCH (n)-[:MENTIONS|CONTAINS_IMAGE|CONTAINS_EVENT|INVOLVES]-(ep:Episode)
                WHERE NOT n:Episode AND NOT n:Session
                RETURN n.id AS id, labels(n) AS labels, n.url AS url, n.embedding AS emb, collect(DISTINCT ep.id) AS ep_ids
            """
            result = session.run(query)
            
            for record in result:
                node_id = record["id"]
                labels = record["labels"]
                ep_ids = record["ep_ids"]
                url = record["url"]
                emb = record["emb"]
                
                # Determine node type
                node_type = "entity"
                if "Image" in labels: 
                    node_type = "image"
                    if url: image_map[node_id] = url
                elif "Event" in labels: 
                    node_type = "event"
                
                # HippoRAG requires file_id to be a comma-separated string
                file_id_str = ",".join(ep_ids)
                
                G.add_node(node_id, type=node_type, id=node_id, file_id=file_id_str)
                node_list.append(node_id)
                
                if emb:
                    node_embeddings_list.append(np.array(emb))
                else:
                    logger.warning(f"Node {node_id} has no embedding!")
                    node_embeddings_list.append(np.zeros(VECTOR_DIM))

            # --- 3. Load Relationships (Edges) ---
            logger.info("[Adapter] Loading Relationships...")
            # Only load semantic edges, for PPR walk in HippoRAG
            # 为了防止查出孤立边，我们限制 head/tail 必须在刚才加载的 node_list 里
            # (虽然 Cypher 传入 huge list 性能一般，但对于 locomo 这种小数据集没问题)
            # 如果数据集很大，可以去掉 IN $node_list 约束，由 NetworkX 自己处理
            rel_query = """
                MATCH (h)-[r]->(t)
                WHERE NOT type(r) IN ['HAS_CHUNK', 'NEXT', 'MENTIONS', 'CONTAINS_IMAGE', 'CONTAINS_EVENT', 'HAS_IMAGE']
                AND h.id IN $node_list AND t.id IN $node_list
                RETURN h.id AS source, type(r) AS rel, t.id AS target, r.embedding AS emb
            """
            result = session.run(rel_query, {'node_list': node_list})
            
            for record in result:
                u, v, r, emb = record["source"], record["target"], record["rel"], record["emb"]
                
                if G.has_node(u) and G.has_node(v):
                    G.add_edge(u, v, relation=r)
                    edge_list.append((u, v))
                    
                    # [修改] 直接读取数据库向量，不再需要 edge_texts
                    if emb:
                        edge_embeddings_list.append(np.array(emb))
                    else:
                        logger.warning(f"Edge {u}-{r}-{v} has no embedding!")
                        edge_embeddings_list.append(np.zeros(VECTOR_DIM))

        # --- 4. Prepare Embeddings ---
        logger.info(f"[Adapter] Finalizing Data (Nodes: {len(node_list)}, Edges: {len(edge_list)})...")
        
        # 节点向量：直接堆叠数据库读出来的
        if node_embeddings_list:
            node_embeddings_matrix = np.array(node_embeddings_list).astype('float32')
            text_embeddings_matrix = np.array(text_embeddings_list).astype('float32')
        else:
            logger.warning("No node embeddings found! HippoRAG will perform poorly.")
            node_embeddings_matrix = np.empty((0, VECTOR_DIM)).astype('float32')
            text_embeddings_matrix = np.array(text_embeddings_list).astype('float32')
        
        if edge_embeddings_list:
            edge_embeddings_matrix = np.array(edge_embeddings_list).astype('float32')
        else:
            logger.warning("No edge embeddings found! HippoRAG will perform poorly.")
            edge_embeddings_matrix = np.empty((0, VECTOR_DIM)).astype('float32')

        # ---- 5. 构建 FAISS 索引 ----
        logger.info("[Adapter] Building FAISS Indices...")
        # Node Index (用于 query2node / ner)
        node_faiss_index = None
        if node_embeddings_matrix.shape[0] > 0:
            # 归一化以支持余弦相似度
            faiss.normalize_L2(node_embeddings_matrix)
            # 使用 Inner Product (IP) 索引
            node_faiss_index = faiss.IndexFlatIP(VECTOR_DIM)
            node_faiss_index.add(node_embeddings_matrix)
        
        # Edge Index (用于 query2edge)
        edge_faiss_index = None
        if edge_embeddings_matrix.shape[0] > 0:
            faiss.normalize_L2(edge_embeddings_matrix)
            edge_faiss_index = faiss.IndexFlatIP(VECTOR_DIM)
            edge_faiss_index.add(edge_embeddings_matrix)

        logger.info("[Adapter] loaded data successfully!")
        
        return {
            "KG": G,
            "text_dict": text_dict,
            "node_list": node_list,
            "edge_list": edge_list,
            "node_embeddings": node_embeddings_matrix,
            "edge_embeddings": edge_embeddings_matrix,
            "image_map": image_map, # 多模态特有
            
            # 这里的 text_embeddings 是用于 Dense Retrieval 的
            # 如果你想用纯向量检索，可以传入 node_embeddings_matrix (对应 Episode 部分)
            "text_embeddings": text_embeddings_matrix, 
            
            "node_faiss_index": node_faiss_index,
            "edge_faiss_index": edge_faiss_index
        }