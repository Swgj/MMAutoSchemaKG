import os
import argparse
import logging
from dotenv import load_dotenv

# 引入组件
from atlas_rag.llm_generator import LLMGenerator
from atlas_rag.retriever.hipporag import HippoRAGRetriever
from atlas_rag.retriever.inference_config import InferenceConfig
from atlas_rag.vectorstore.embedding_model import EmbeddingAPI
from atlas_rag.multimodal.hipporag_adapter import Neo4jToHippoAdapter
from openai import OpenAI

logger = logging.getLogger(__name__)
load_dotenv()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, default="What did Caroline say about the necklace?")
    parser.add_argument("--uri", type=str, default="bolt://localhost:7687")
    parser.add_argument("--user", type=str, default="neo4j")
    parser.add_argument("--password", type=str, default="password")
    args = parser.parse_args()

    api_key = os.getenv("GEMINI_API_KEY") # 或 OPENAI_API_KEY
    base_url = os.getenv("GEMINI_BASE_URL")

    # 1. 初始化模型
    client = OpenAI(api_key=api_key, base_url=base_url)
    llm_generator = LLMGenerator(client, model_name="gemini-2.5-flash") # 用于 HippoRAG 内部的 NER
    embedding_model = EmbeddingAPI(client, model_name="gemini-embedding-001")
    
    # 2. 从 Neo4j 加载数据
    adapter = Neo4jToHippoAdapter(args.uri, args.user, args.password, embedding_model)
    data = adapter.load_data()
    adapter.close()
    
    # 3. 配置检索参数
    config = InferenceConfig()
    config.hipporag_mode = "query2node" # 推荐模式：Query -> 相似边 -> PPR
    config.topk_nodes = 20              # PPR 初始注入节点数
    config.topk_edges = 10              # Query 匹配多少条边
    config.ppr_alpha = 0.85             # PageRank 阻尼系数
    
    # 4. 初始化检索器
    logger.info("\n🔍 Initializing HippoRAG Retriever...")
    retriever = HippoRAGRetriever(
        llm_generator=llm_generator,
        sentence_encoder=embedding_model,
        data=data,
        inference_config=config
    )
    
    # 5. 执行检索
    logger.info(f"\n❓ Query: {args.query}")
    passages, passage_ids = retriever.retrieve(args.query, topN=3)
    
    # 6. 展示结果 (多模态)
    image_map = data.get("image_map", {})
    
    logger.info("\n✅ Search Results:")
    for i, (content, pid) in enumerate(zip(passages, passage_ids)):
        logger.info(f"\n--- Result {i+1} (Chunk ID: {pid}) ---")
        logger.info(f"📄 Text: {content[:150]}...")
        
        # 检查这个 Chunk 里有没有图片
        # 简单的字符串匹配，或者你可以去 Neo4j 查
        if "(Image:" in content:
            logger.info("🖼️  Images found in transcript:")
            # 提取 Image ID 并查找 URL
            import re
            img_ids = re.findall(r'\(Image: (IMG_.*?)\)', content)
            for img_id in img_ids:
                url = image_map.get(img_id, "Unknown URL")
                logger.info(f"   - {img_id}: {url[:50]}...")

if __name__ == "__main__":
     # setting logger, print info in console
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    # filter httpx info
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logger.info("Starting HippoRAG Test")

    main()