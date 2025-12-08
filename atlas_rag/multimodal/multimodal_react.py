
import re
from typing import Union, Tuple, Dict, List, Optional, Set
from logging import Logger

from atlas_rag.llm_generator import LLMGenerator
from atlas_rag.retriever.base import BaseEdgeRetriever, BasePassageRetriever
from atlas_rag.multimodal.utils import load_image_content_from_dict


class MultimodalReAct:
    def __init__(self, llm: LLMGenerator):
        self.llm = llm

    def generate_with_rag_react(
        self,
        question: str,
        retriever: Union[BaseEdgeRetriever, BasePassageRetriever],
        image_map: Dict[str, str],
        max_iterations: int = 5,
        max_new_tokens: int = 1024,
        logger: Logger = None,
        ) -> Tuple[str, List[Tuple[str, str, str]]]:
        search_history: List[Tuple[str, str, str]] = []
        current_context, initial_fragments = self._build_initial_context(question, retriever)
        # 使用局部 set 记录已经展示过的文本片段和图片 ID，作用域仅限本次调用
        seen_fragments: Set[str] = set(initial_fragments)
        seen_image_ids: Set[str] = set()

        try:
            for iteration in range(max_iterations):
                analysis_response = self.llm.generate_with_multimodal_react(
                    question=question,
                    context=current_context,
                    max_new_tokens=max_new_tokens,
                    search_history=search_history,
                    logger=logger,
                )
                if logger:
                    logger.info(f"[ReAct] Iteration {iteration} response: {analysis_response}")

                try:
                    thought, action, answer = self._parse_analysis_response(analysis_response)
                except ValueError:
                    if logger:
                        logger.warning("Failed to parse ReAct response, stopping early.")
                    break

                if logger:
                    logger.info(f"[ReAct] Thought={thought} action={action} answer={answer}")

                if not self._needs_more_info(answer):
                    search_history.append((thought, action, "Answered with current context"))
                    return answer, search_history

                next_context, observation, seen_fragments, seen_image_ids = self._plan_next_context(
                    action,
                    current_context,
                    retriever,
                    image_map,
                    seen_fragments,
                    seen_image_ids,
                    logger,
                )
                search_history.append((thought, action, observation))
                current_context = next_context
        finally:
            # 显式清空局部 set，帮助及时释放引用（尽管函数返回后也会被 GC）
            seen_fragments.clear()
            seen_image_ids.clear()

        return "Unable to obtain a confident answer within iteration limit.", search_history

    def _build_initial_context(
        self,
        question: str,
        retriever: Union[BaseEdgeRetriever, BasePassageRetriever],
    ) -> Tuple[List[Dict[str, str]], List[str]]:
        fragments: List[str] = []
        if isinstance(retriever, BaseEdgeRetriever):
            fragments, _ = retriever.retrieve(question, topN=5)
        elif isinstance(retriever, BasePassageRetriever):
            retrieved, retrieved_ids = retriever.retrieve(question, topN=5)
            fragments = [f"\n[Fragment {text_id}]\n {text}" for text_id, text in zip(retrieved_ids, retrieved)]
        return self._text_fragments_to_context(fragments), retrieved_ids

    def _text_fragments_to_context(self, fragments: List[str]) -> List[Dict[str, str]]:
        return [{"type": "text", "text": fragment} for fragment in fragments if fragment]

    def _needs_more_info(self, answer: str) -> bool:
        return "need more information" in answer.lower()

    def _plan_next_context(
        self,
        action: str,
        current_context: List[Dict[str, str]],
        retriever: Union[BaseEdgeRetriever, BasePassageRetriever],
        image_map: Dict[str, str],
        seen_fragments: Set[str],
        seen_image_ids: Set[str],
        logger: Optional[Logger],
    ) -> Tuple[List[Dict[str, str]], str, Set[str], Set[str]]:
        if not action:
            if logger:
                logger.info("No action parsed from ReAct response.")
            return list(current_context), "No action provided in ReAct response.", seen_fragments, seen_image_ids

        action_lower = action.lower()
        next_context = list(current_context)

        if "inspect image" in action_lower:
            img_ids = re.findall(r"(IMG_[a-zA-Z0-9_]+)", action)
            # 过滤掉已经展示过的图片 ID
            img_ids = [id for id in img_ids if id not in seen_image_ids]
            seen_image_ids.update(img_ids)
            image_parts = self._load_images(img_ids, image_map)
            if image_parts:
                next_context.extend(image_parts)
                if logger:
                    logger.info(f"Loaded images for ids: {img_ids}")
                return next_context, f"Loaded and presented images: {img_ids}", seen_fragments, seen_image_ids
            return next_context, f"Could not locate images for IDs: {img_ids}", seen_fragments, seen_image_ids

        if "search for" in action_lower:
            query = action.lower().split("search for", 1)[1].strip(" []\"'")

            new_fragments, _ = retriever.retrieve(query, topN=3)
            if new_fragments:
                # 只保留本次新出现的片段
                new_fragments = [frag for frag in new_fragments if frag not in seen_fragments]
                seen_fragments.update(new_fragments)
                next_context.extend(self._text_fragments_to_context(new_fragments))
                if logger:
                    logger.info(f"Retrieved {len(new_fragments)} fragments for '{query}'")
                return next_context, f"Retrieved {len(new_fragments)} new fragments for '{query}'.", seen_fragments, seen_image_ids
            return next_context, f"No new fragments found for '{query}'.", seen_fragments, seen_image_ids

        return next_context, "No valid action parsed.", seen_fragments, seen_image_ids

    def _load_images(self, image_ids: List[str], image_map: Dict[str, str]) -> List[Dict[str, str]]:
        content_parts: List[Dict[str, str]] = []
        for image_id in image_ids:
            loaded = load_image_content_from_dict(image_id, image_map)
            if loaded and isinstance(loaded.get("content"), list):
                content_parts.extend(
                    [part for part in loaded["content"] if isinstance(part, dict)]
                )
        return content_parts

    def _parse_analysis_response(self, analysis_response: str) -> Tuple[str, str, str]:
        if "Action:" not in analysis_response and "Answer:" not in analysis_response:
            raise ValueError("Incomplete ReAct response")

        thought = self._extract_section(analysis_response, "Thought") or "No thought provided"
        action = self._extract_section(analysis_response, "Action")
        answer = self._extract_section(analysis_response, "Answer", raw=True) or "Need more information"
        return thought, action, answer

    def _extract_section(self, text: str, label: str, raw: bool = False) -> str:
        parts = text.split(f"{label}:", 1)
        if len(parts) < 2:
            return ""
        section = parts[1].strip()
        if raw:
            return section
        return section.split("\n", 1)[0].strip()
