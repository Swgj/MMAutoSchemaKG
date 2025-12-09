import os
import argparse
import logging
import pandas as pd
import yaml
import types
import concurrent.futures
from datetime import datetime
from tqdm import tqdm
from pandas import DataFrame, Series
from typing import Tuple, List, Dict
from dotenv import load_dotenv
from openai import OpenAI
from contextvars import ContextVar

from atlas_rag.llm_generator import LLMGenerator
from atlas_rag.retriever.hipporag import HippoRAGRetriever
from atlas_rag.vectorstore.embedding_model import EmbeddingAPI
from atlas_rag.multimodal.hipporag_adapter import Neo4jToHippoAdapter
from atlas_rag.retriever.inference_config import InferenceConfig
from atlas_rag.multimodal.multimodal_react import MultimodalReAct


logger = logging.getLogger(__name__)

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
base_url = os.getenv("GEMINI_BASE_URL")

row_context: ContextVar[str] = ContextVar('row_context', default='')
class ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.row_id = row_context.get()
        return True

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


def process_single_row(
    index: str, row: Series,
    llm_generator: LLMGenerator,
    retriever: HippoRAGRetriever,
    image_map: dict,
    max_iterations: int,
    max_new_tokens: int,
) -> Tuple[str, str, List[tuple]]:
    token = row_context.set(f"Row-{index}")
    try:
        question = row.get('question')
        if not question:
            return index, "Error: Empty question", []
        
        mm_react = MultimodalReAct(llm_generator)
        answer, history = mm_react.generate_with_rag_react(
            question=question,
            retriever=retriever,
            image_map=image_map,
            max_iterations=max_iterations,
            max_new_tokens=max_new_tokens,
            logger=logger,
        )
        return index, answer, history
    except Exception as e:
        logger.error(f"Error processing row {index}: {str(e)}")
        return index, f"Error: {str(e)}", []
    finally:
        row_context.reset(token)

def judge_answer(index: str, question: str, answer: str, reference_answer: str, llm_judger: LLMGenerator) -> bool:
    """
    Judge if the answer is semantically consistent with the reference answer.
    """
    token = row_context.set(f"Eval-{index}")

    system_prompt = """
    You are an impartial judge evaluating the correctness of an AI-generated answer.
    
    YOUR TASK:
    Compare the [Generated Answer] with the [Reference Answer] based on the [Question].
    
    JUDGMENT CRITERIA:
    1. **Semantic Equivalence**: If the generated answer contains the correct information/key entities present in the reference, mark it as TRUE.
    2. **Flexibility**: Ignore differences in phrasing, punctuation, or length, provided the core meaning is preserved.
    3. **Hallucinations**: If the generated answer contains contradictory information to the reference, mark it as FALSE.
    
    OUTPUT FORMAT:
    First, provide a brief reasoning (1 sentence).
    Then, end with exactly "JUDGMENT: TRUE" or "JUDGMENT: FALSE".
    """

    user_prompt = f"""
    [Question]: {question}
    
    [Reference Answer]: {reference_answer}
    
    [Generated Answer]: {answer}
    """
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    
    try:
        response = llm_judger.generate_response(messages)
    
        response_upper = response.upper()
        
        if "JUDGMENT: TRUE" in response_upper:
            res = True
        elif "JUDGMENT: FALSE" in response_upper:
            res = False
        else:
            res = "TRUE" in response_upper

        return index, res
    finally:
        row_context.reset(token)

def main(args = None):

    if not args:
        parser = argparse.ArgumentParser(description="Debug Multimodal ReAct + HippoRAG pipeline.")
        # KG Database
        parser.add_argument("--neo4j_uri", type=str, default="bolt://localhost:7687")
        parser.add_argument("--neo4j_user", type=str, default="neo4j")
        parser.add_argument("--neo4j_password", type=str, default="password")

        # LLM and Embedding Model
        parser.add_argument("--model_name", type=str, default="gemini-2.5-flash", help="LLM model name")
        parser.add_argument("--embedding_model_name", type=str, default="gemini-embedding-001", help="Embedding model name")
        
        # HippoRAG
        parser.add_argument("--hipporag_mode", type=str, default="query2node", choices=["query2edge", "query2node"])
        parser.add_argument("--topk_nodes", type=int, default=20)
        parser.add_argument("--ppr_alpha", type=float, default=0.85)

        # ReAct
        parser.add_argument("--max_iterations", type=int, default=5)
        parser.add_argument("--max_new_tokens", type=int, default=1024)

        # Data
        parser.add_argument("--data_path", type=str)
        parser.add_argument("--output_path", type=str)
        parser.add_argument("--max_workers", type=int, default=1)

        args = parser.parse_args()
    else:
        args = types.SimpleNamespace(**args)
    
    # 1. Load data
    dataset = pd.read_json(args.data_path)
    dataset["rag_answer"] = None
    dataset["rag_history"] = None
    dataset["rag_judgment"] = None

    # 2. Build resources
    llm_generator, retriever, data = build_resources(
        neo4j_uri=args.neo4j_uri,
        neo4j_user=args.neo4j_user,
        neo4j_password=args.neo4j_password,
        model_name=args.model_name,
        embedding_model_name=args.embedding_model_name,
        hipporag_mode=args.hipporag_mode,
        topk_nodes=args.topk_nodes,
        ppr_alpha=args.ppr_alpha,
    )
    

    # 3. Multimodal ReAct, parallel processing
    logger.info(f"Processing {len(dataset)} rows with {args.max_workers} workers")

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = []
        for index, row in dataset.iterrows():
            future = executor.submit(
                process_single_row,
                index=index,
                row=row,
                llm_generator=llm_generator,
                retriever=retriever,
                image_map=data.get("image_map", {}),
                max_iterations=args.max_iterations,
                max_new_tokens=args.max_new_tokens,
            )
            futures.append(future)

        for future in tqdm(concurrent.futures.as_completed(futures), desc="Processing rows", total=len(futures)):
            try:
                index, answer, history = future.result()
                dataset.at[index, "rag_answer"] = answer
                dataset.at[index, "rag_history"] = history
            except Exception as e:
                logger.error(f"Error processing row {index}: {str(e)}")
                dataset.at[index, "rag_answer"] = f"Error: {str(e)}"
                dataset.at[index, "rag_history"] = []
    
    dataset.to_json(args.output_path, orient="records", force_ascii=False, indent=4)
    

    # 4. Evaluate
    logger.info(f"Evaluating {len(dataset)} rows")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = []
        for index, row in dataset.iterrows():
            future = executor.submit(
                judge_answer,
                index=index,
                question=row.get('question'),
                answer=row.get('rag_answer'),
                reference_answer=row.get('answer'),
                llm_judger=llm_generator,
            )
            futures.append(future)

        for future in tqdm(concurrent.futures.as_completed(futures), desc="Evaluating rows", total=len(futures)):
            try:
                index, judgment = future.result()
                dataset.at[index, "rag_judgment"] = judgment
            except Exception as e:
                logger.error(f"Error evaluating row {index}: {str(e)}")
    dataset.to_json(args.output_path, orient="records", force_ascii=False, indent=4)

    # ===== Calculate Metrics =====
    acc = dataset["rag_judgment"].mean()
    logger.info(f"Accuracy: {acc}")


if __name__ == "__main__":
    args = yaml.safe_load(open("atlas_rag/evaluation/locomo_hard.yml", "r"))
    os.makedirs("logs", exist_ok=True)
    
    # logger output to file
    log_format = logging.Formatter("%(asctime)s - [%(row_id)s] - %(name)s - %(levelname)s - %(message)s")

    file_hanlder = logging.FileHandler(f"logs/mm_locomo_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    file_hanlder.setLevel(logging.INFO)
    file_hanlder.setFormatter(log_format)
    file_hanlder.addFilter(ContextFilter())

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.ERROR)
    console_handler.setFormatter(log_format)
    console_handler.addFilter(ContextFilter())

    logging.basicConfig(
        level=logging.INFO,
        handlers=[file_hanlder, console_handler],
        force=True,
    )

    row_context.set("Main")
    logger.info(f"Config: {args}")

    main(args)