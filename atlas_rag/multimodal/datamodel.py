from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from atlas_rag.multimodal.utils import url_to_base64
import logging

logger = logging.getLogger(__name__)

@dataclass
class MultimodalMessage:
    """single multimodal message"""
    role: str
    content: str
    img_url: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MultimodalMessage":
        """create a MultimodalMessage from a dictionary"""
        return cls(
            role=data.get("role"),
            content=data.get("content"),
            img_url=data.get("img_url", []),
            metadata=data.get("metadata", {})
        )
    
    def format_to_openai_content(self, img_start_index: int = 0, id_prefix: str = "") -> Dict[str, Any]:
        """format the message to OpenAI format"""
        content_list = []
        image_map = {}  # UUID -> image url
        transcript_parts = []
        cur_img_idx = img_start_index

        # 1. text content
        # for LLM-readable content
        text = {
            "type": "text",
            "text": f"\n**{self.role}**: {self.content}\n"
        }
        content_list.append(text)

        # for Human-readable transcript
        transcript_parts.append(f"[{self.role}]: {self.content}")

        # 2. image content
        if self.img_url:
            for img in self.img_url:
                # Entity Name for the image node
                img_node_id = f"IMG_{id_prefix}_{cur_img_idx}"

                # for LLM-readable content
                content_list.append({
                    "type": "text",
                    "text": f"(Image uploaded by {self.role}: <{img_node_id}> following:)\n"
                })

                base64_img = url_to_base64(img)
                if base64_img is not None:
                    content_list.append({
                        "type": "image_url",
                        "image_url": {
                            "url": base64_img
                        }
                    })
                else:
                    logger.warning(f"Warning: Failed to convert image {img} to base64")

                # for Human-readable transcript
                transcript_parts.append(f"(Image: {img_node_id})")

                image_map[img_node_id] = img
                cur_img_idx += 1
        
        return {
            "content_list": content_list,
            "image_map": image_map,
            "next_img_idx": cur_img_idx,
            "transcript": "\n".join(transcript_parts),
        }

@dataclass
class ConversationSession:
    """a complete conversation session"""
    id: str
    messages: List[MultimodalMessage]
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationSession":
        """create a ConversationSession from a dictionary"""
        return cls(
            id=data.get("id"),
            messages=[MultimodalMessage.from_dict(msg) for msg in data.get("messages", [])],
            metadata=data.get("metadata", {})
        )


@dataclass
class ProcessedChunk:
    """a chunk from conversation session, for KG constraction"""
    chunk_id: str   # unique id for the chunk
    original_id: str    # id of the original conversation session
    window_index: int
    transcript: str
    llm_payload: List[Dict[str, Any]]   #OpenAI format Multimodal Message list
    image_map: Dict[str, str]   # map of image id to image url
    session_metadata: Dict[str, Any] = field(default_factory=dict) # Store the original session metadata
