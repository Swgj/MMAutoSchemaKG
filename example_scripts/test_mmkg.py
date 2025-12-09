import os
import time
import argparse
import logging
from configparser import ConfigParser
from openai import OpenAI
from dotenv import load_dotenv

# 引入原项目组件
from atlas_rag.llm_generator import LLMGenerator
from atlas_rag.kg_construction.triple_config import ProcessingConfig

# 引入我们构建的多模态抽取器
from atlas_rag.multimodal.extraction import MultimodalKGExtractor

load_dotenv()
logger = logging.getLogger(__name__)


def main():
    # 1. 参数解析
    parser = argparse.ArgumentParser(description="Run Multimodal KG Extraction")
    parser.add_argument("--data_dir", type=str, default="example_data/locomo_hard_data/", help="Directory containing the json data")
    parser.add_argument("--filename", type=str, default="locomo_hard_0", help="Filename without extension (e.g. locomo_hard_0)")
    parser.add_argument("--model", type=str, default="gemini-2.5-flash", help="Model name (must support vision)")
    parser.add_argument("--api_key", type=str, default=os.environ.get("GEMINI_API_KEY"), help="API Key")
    parser.add_argument("--base_url", type=str, default=os.environ.get("GEMINI_BASE_URL"), help="Custom API Base URL")
    args = parser.parse_args()

    # 2. 初始化 OpenAI 客户端
    # 这一步非常关键，确保您的 Client 指向了正确的服务商
    if not args.api_key:
        print("⚠️ Warning: No API Key provided. Please set OPENAI_API_KEY env var or pass --api_key")

    client = OpenAI(
        api_key=args.api_key,
        base_url=args.base_url
    )

    # 3. 初始化 LLM 生成器
    # max_workers 控制并发数，多模态请求较大，建议不要设太大以免触发 Rate Limit
    triple_generator = LLMGenerator(client, model_name=args.model, max_workers=8)

    # 4. 配置抽取参数
    # 这里我们将 window_size 和 window_overlap 注入到 Config 中
    kg_config = ProcessingConfig(
        model_path=args.model,
        data_directory=args.data_dir,
        filename_pattern=args.filename,
        output_directory=f"./generation_result/{args.model}", # 结果输出位置
        batch_size_triple=5,   # 多模态 Payload 很大，Batch Size 建议调小 (5-10)
        max_new_tokens=4096,
        record=True,           # 记录 Token 消耗
        # debug_mode= True,      # 调试模式，只处理前20个样本
        # --- 多模态特有参数 (会被 getattr 读取) ---
        # window_size=10, 
        # window_overlap=2
    )
    # 手动补丁：因为 ProcessingConfig 是 dataclass，可能不支持直接传未知参数
    # 我们手动绑上去，MultimodalDataProcessor 会用 getattr 读取
    kg_config.window_size = 10
    kg_config.window_overlap = 2

    # 5. 启动抽取
    print(f"🚀 [Start] Extracting KG from {args.filename} using {args.model}...")
    start_time = time.time()

    extractor = MultimodalKGExtractor(model=triple_generator, config=kg_config)
    extractor.run_extraction()

    total_time = time.time() - start_time
    print(f"🎉 [Done] Total time: {total_time:.2f} seconds")


if __name__ == "__main__":
    # setting logger, print info in console
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    # filter httpx info
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logger.info("Starting Multimodal KG Extraction")

    main()