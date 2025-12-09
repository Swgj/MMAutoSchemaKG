import os
import argparse
import logging
from typing import Tuple, List

from dotenv import load_dotenv
from openai import OpenAI

from atlas_rag.llm_generator import LLMGenerator
from atlas_rag.vectorstore.embedding_model import EmbeddingAPI
from atlas_rag.multimodal.hipporag_adapter import Neo4jToHippoAdapter
from atlas_rag.retriever.hipporag import HippoRAGRetriever
from atlas_rag.retriever.inference_config import InferenceConfig
from atlas_rag.multimodal.multimodal_react import MultimodalReAct


logger = logging.getLogger("mmrag_debug")


def build_resources(
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    model_name: str,
    embedding_model_name: str,
    hipporag_mode: str,
    topk_nodes: int,
    ppr_alpha: float,
) -> Tuple[LLMGenerator, HippoRAGRetriever, dict]:
    """
    加载 LLM、Embedding 模型和图数据，返回 (llm_generator, retriever, data_dict)。
    """
    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")
    base_url = os.getenv("GEMINI_BASE_URL")

    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not found! Please set it in your environment or .env file.")

    # 1. 初始化 OpenAI Client / Embedding / LLM
    client = OpenAI(api_key=api_key, base_url=base_url)
    embedding_model = EmbeddingAPI(client, model_name=embedding_model_name)
    llm_generator = LLMGenerator(client, model_name=model_name)

    # 2. 从 Neo4j 加载 HippoRAG 所需的数据结构
    adapter = Neo4jToHippoAdapter(neo4j_uri, neo4j_user, neo4j_password, embedding_model)
    data = adapter.load_data()
    adapter.close()

    # 3. 配置 HippoRAG
    config = InferenceConfig()
    config.hipporag_mode = hipporag_mode
    config.topk_nodes = topk_nodes
    config.ppr_alpha = ppr_alpha

    retriever = HippoRAGRetriever(
        llm_generator=llm_generator,
        sentence_encoder=embedding_model,
        data=data,
        inference_config=config,
        logger=logger,
    )

    return llm_generator, retriever, data


def run_multimodal_react_debug(
    question: str,
    llm_generator: LLMGenerator,
    retriever: HippoRAGRetriever,
    image_map: dict,
    max_iterations: int,
    max_new_tokens: int,
) -> Tuple[str, List[tuple]]:
    """
    使用 MultimodalReAct + HippoRAGRetriever 执行一次多轮 ReAct 推理，并返回答案和搜索历史。
    """
    mm_react = MultimodalReAct(llm_generator)
    answer, history = mm_react.generate_with_rag_react(
        question=question,
        retriever=retriever,
        image_map=image_map,
        max_iterations=max_iterations,
        max_new_tokens=max_new_tokens,
        logger=logger,
    )
    return answer, history


def main():
    parser = argparse.ArgumentParser(description="Debug Multimodal ReAct + HippoRAG pipeline.")
    parser.add_argument("--neo4j_uri", type=str, default="bolt://localhost:7687")
    parser.add_argument("--neo4j_user", type=str, default="neo4j")
    parser.add_argument("--neo4j_password", type=str, default="password")

    parser.add_argument("--model_name", type=str, default="gemini-2.5-flash", help="LLM model name")
    parser.add_argument("--embedding_model_name", type=str, default="gemini-embedding-001", help="Embedding model name")
    parser.add_argument("--hipporag_mode", type=str, default="query2node", choices=["query2edge", "query2node"])
    parser.add_argument("--topk_nodes", type=int, default=20)
    parser.add_argument("--ppr_alpha", type=float, default=0.85)

    parser.add_argument("--max_iterations", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=1024)

    parser.add_argument(
        "--question",
        type=str,
        default="In the pride event mentioned by Caroline, there is an image. What pattern is displayed on the large umbrella held overhead?",
        help="Question to ask the multimodal KG.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger.info("Building resources for Multimodal ReAct debug run...")
    llm_gen, retriever, data = build_resources(
        neo4j_uri=args.neo4j_uri,
        neo4j_user=args.neo4j_user,
        neo4j_password=args.neo4j_password,
        model_name=args.model_name,
        embedding_model_name=args.embedding_model_name,
        hipporag_mode=args.hipporag_mode,
        topk_nodes=args.topk_nodes,
        ppr_alpha=args.ppr_alpha,
    )
    image_map = data.get("image_map", {})

    logger.info("Starting Multimodal ReAct reasoning...")
    answer, history = run_multimodal_react_debug(
        question=args.question,
        llm_generator=llm_gen,
        retriever=retriever,
        image_map=image_map,
        max_iterations=args.max_iterations,
        max_new_tokens=args.max_new_tokens,
    )

    print("\n==================== FINAL ANSWER ====================\n")
    print(answer)
    print("\n==================== SEARCH HISTORY ====================\n")
    for i, (thought, action, observation) in enumerate(history):
        print(f"[Step {i+1}]")
        print(f"Thought     : {thought}")
        print(f"Action      : {action}")
        print(f"Observation : {observation}")
        print("------------------------------------------------------")


if __name__ == "__main__":
    main()

