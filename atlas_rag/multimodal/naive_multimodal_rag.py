from atlas_rag.llm_generator import LLMGenerator
from atlas_rag.retriever.simple_retriever import SimpleTextRetriever
from atlas_rag.multimodal.utils import load_image_content_from_dict
from atlas_rag.llm_generator.prompt.rag_prompt import rag_qa_system as RAG_QA_SYSTEM

from typing import Dict, List, Tuple
from logging import Logger

import re


class NaiveMultimodalRAG:
    def __init__(self, llm: LLMGenerator):
        self.llm = llm
    
    def generate_with_rag(
        self, 
        question: str,
        retriever: SimpleTextRetriever,
        image_map: Dict[str, str],
        max_new_tokens: int = 1024,
        logger: Logger = None,
    ) -> Tuple[str, List[tuple]]:
        search_history: List[tuple] = []

        # retrieve the top k passages
        topk_passage, topk_passage_ids = retriever.retrieve(question, topN=3)
        fragments = [f"\n[Fragment {text_id}]\n {text}" for text_id, text in zip(topk_passage_ids, topk_passage)]

        search_history.append(("Retrieved passages", topk_passage_ids))
        
        user_context = self._text_fragments_to_multimodal_context(fragments, image_map)
        
        final_user_content = []
        final_user_content.append(
            {
                "type": "text",
                "text": "--- Current Context (Text & Images) ---",
            }
        )
        final_user_content.extend(user_context)
        final_user_content.append(
            {
                "type": "text",
                "text": (
                    "\n--- Question ---\n"
                    f"{question}\n\nPlease think step by step and output Thought / Action / Answer."
                ),
            }
        )

        messages = [
            {"role": "system", "content": RAG_QA_SYSTEM},
            {"role": "user", "content": final_user_content}
        ]

        analysis_response = self.llm.generate_response(messages, max_new_tokens=max_new_tokens)
        thought = self._extract_section(analysis_response, "Thought") or "No thought provided"
        answer = self._extract_section(analysis_response, "Answer", raw=True) or "Need more information"
        
        logger.info(f"Thought: {thought}")
        logger.info(f"Answer: {answer}")
        
        return answer, search_history

    def _text_fragments_to_multimodal_context(
        self,
        fragments: List[str],
        image_map: Dict[str, str]
    ) -> List[Dict[str, str]]:
        """
        Convert text fragments to multimodal context.
        """
        context = []
        for fragment in fragments:
            context.append({
                "type": "text",
                "text": fragment
            })
            img_ids = re.findall(r"(IMG_[a-zA-Z0-9_]+)", fragment)
            for img_id in img_ids:
                loaded = load_image_content_from_dict(img_id, image_map)
                if loaded and isinstance(loaded.get("content"), list):
                    context.extend(
                        [part for part in loaded["content"] if isinstance(part, dict)]
                    )
        return context
    
    def _extract_section(self, text: str, label: str, raw: bool = False) -> str:
        parts = text.split(f"{label}:", 1)
        if len(parts) < 2:
            return ""
        section = parts[1].strip()
        if raw:
            return section
        return section.split("\n", 1)[0].strip()