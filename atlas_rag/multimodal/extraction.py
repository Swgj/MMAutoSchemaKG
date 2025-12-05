"""
Multimodal Knowledge Graph Extraction
"""

from typing import List, Dict, Any
from atlas_rag.kg_construction.triple_config import ProcessingConfig
from atlas_rag.kg_construction.triple_extraction import KnowledgeGraphExtractor
from atlas_rag.llm_generator.llm_generator import LLMGenerator
from atlas_rag.llm_generator.prompt.mmkg_prompt import MULTIMODAL_TRIPLE_INSTRUCTIONS
from atlas_rag.llm_generator.format.validate_json_schema import ATLAS_SCHEMA
from atlas_rag.multimodal.chunk_splitter import ConversationSplitter
from atlas_rag.multimodal.datamodel import ConversationSession, ProcessedChunk
from tqdm import tqdm
import logging
import os
import json
import torch

RESULT_SCHEMA = ATLAS_SCHEMA

logger = logging.getLogger(__name__)

class MultimodalDataProcessor:
    """
    Process multimodal data for knowledge graph extraction.
    Split the data into chunks
    """
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.window_size = getattr(config, "window_size", 10)
        self.window_overlap = getattr(config, "window_overlap", 2)
        self.splitter = ConversationSplitter(self.window_size, self.window_overlap)
    
    
    def prepare_dataset(self, raw_dataset) -> List[ProcessedChunk]:
        """
        Load the raw dataset(formatted json) and split it into chunks
        """
        processed_chunks: List[ProcessedChunk] = []

        for single_session in raw_dataset:
            session = ConversationSession.from_dict(single_session)
            chunks = self.splitter.split_messages(session)
            processed_chunks.extend(chunks)
            logger.info(f"Split session {session.id} into {len(chunks)} chunks")
        logger.info(f"Total {len(processed_chunks)} chunks prepared")
        return processed_chunks


class MultimodalDataLoader:
    """
    Batch generator that constructs the final prompt for the LLM.
    """
    def __init__(self, chunks: List[ProcessedChunk], config: ProcessingConfig):
        self.chunks = chunks
        self.config = config
        
    def create_batch_instructions(self, batch_chunks: List[ProcessedChunk]) -> Dict[str, List[Any]]:
        messages_dict = {key: [] for key in RESULT_SCHEMA.keys()}
        lang = "en" 
        system_msg = MULTIMODAL_TRIPLE_INSTRUCTIONS.get(lang, MULTIMODAL_TRIPLE_INSTRUCTIONS["en"])['system']

        for chunk in batch_chunks:
            multimodal_content = chunk.llm_payload
            for key in RESULT_SCHEMA.keys():
                task_instruction = MULTIMODAL_TRIPLE_INSTRUCTIONS.get(lang, MULTIMODAL_TRIPLE_INSTRUCTIONS["en"])[key]
                
                # 组装 User Message
                user_content_list = [{
                    "type": "text",
                    "text": f"{task_instruction}\n\nHere is the conversation content:\n"
                }]
                user_content_list.extend(multimodal_content)

                messages_dict[key].append([
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_content_list} 
                ])
        return messages_dict

    def __iter__(self):
        batch_size = self.config.batch_size_triple
        start_idx = self.config.resume_from * batch_size
        for i in range(start_idx, len(self.chunks), batch_size):
            batch = self.chunks[i : i + batch_size]
            instructions = self.create_batch_instructions(batch)
            yield instructions, batch


class MultimodalKGExtractor(KnowledgeGraphExtractor):
    """
    Multimodal implementation of KG Extractor.
    Inherits common logic (file naming, result parsing, etc.) from KnowledgeGraphExtractor.
    """
    def __init__(self, model: LLMGenerator, config: ProcessingConfig):
        super().__init__(model, config)
        

    def load_dataset(self) -> List[Dict]:
        """Override: Load JSON data from file directly (instead of using 'datasets' library)"""
        data_path = os.path.join(self.config.data_directory, self.config.filename_pattern + ".json")
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading dataset from {data_path}: {e}")
            return []

    def run_extraction(self):
        """Override: Main execution loop adapted for Multimodal Data"""
        logger.info(f"Starting Multimodal KG Extraction using {self.config.model_path}")
        
        # 1. 准备数据 (使用多模态 Processor)
        raw_data = self.load_dataset()
        if not raw_data: return

        processor = MultimodalDataProcessor(self.config)
        chunks = processor.prepare_dataset(raw_data)
        
        if not chunks:
            logger.error("No chunks to process.")
            return

        loader = MultimodalDataLoader(chunks, self.config)
        
        # 2. 准备输出文件 
        output_file = self.create_output_filename()
        
        batch_counter = 0
        total_batches = (len(chunks) + self.config.batch_size_triple - 1) // self.config.batch_size_triple

        with torch.no_grad():
            with open(output_file, "w", encoding="utf-8") as output_stream:
                
                for instructions_dict, batch_chunks in tqdm(loader, total=total_batches, desc="Extracting Batches"):
                    batch_counter += 1
                    stage_outputs = {}
                    
                    # 3. 执行抽取 (复用父类 process_stage!)
                    # 这极大地简化了代码，自动处理了 LLM 调用、重试和 JSON 解析
                    result_keys = list(RESULT_SCHEMA.keys())
                    for key in result_keys:
                        # 传入多模态的 instructions 和 schema
                        batch_result = self.process_stage(instructions_dict[key], result_schema=RESULT_SCHEMA[key])
                        stage_outputs[key] = batch_result

                    # 4. 写入结果
                    # 这里我们需要手动组装结果，因为 prepare_result_dict 的参数不太匹配多模态特有的 image_map
                    for i in range(len(batch_chunks)):
                        chunk = batch_chunks[i]
                        
                        # 构造基础结果
                        result_entry = {
                            "id": chunk.chunk_id,
                            "original_id": chunk.original_id,
                            "chunk_index": chunk.window_index,
                            "transcript": chunk.transcript,
                            "image_map": chunk.image_map, # 多模态特有字段
                            "metadata": {} 
                        }

                        # 填充各阶段结果
                        for key in result_keys:
                            llm_output, triple_dict = stage_outputs[key]
                            # process_stage 返回的是 (raw_output_list, parsed_dict_list)
                            # 如果 record=True, raw_output 是 tuple(text, usage)
                            
                            # 处理 raw output
                            if self.config.record and isinstance(llm_output[i], (list, tuple)):
                                result_entry[f"{key}_output"] = llm_output[i][0]
                                result_entry[f"{key}_usage"] = llm_output[i][1]
                            else:
                                result_entry[f"{key}_output"] = llm_output[i]
                                
                            # 处理 parsed dict
                            result_entry[f"{key}_dict"] = triple_dict[i]

                        # 写入文件
                        output_stream.write(json.dumps(result_entry, ensure_ascii=False) + "\n")
                        output_stream.flush()

        logger.info(f"Extraction complete! Results saved to {output_file}")