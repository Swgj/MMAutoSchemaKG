"""
Conversation Chunk Splitter
"""

from typing import List, Dict, Any
from atlas_rag.llm_generator.prompt.mmkg_prompt import conversation_prefix
import copy

class ConversationSplitter:
    """
    Split a long conversation into windows of a fixed size.(Chunks)
    """
    
    def __init__(self, window_size: int = 10, window_overlap: int = 2):
        """
        Initialize Splitter
        
        Args:
            window_size (int): The number of messages in each window (Message count)
            window_overlap (int): The number of messages in the overlap between windows (Preserve context continuity)
        """
        self.window_size = window_size
        self.window_overlap = window_overlap


    def split_conversation(self, conversation_id: str, conversation: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Split conversation into chunks"""
        if not conversation:
            return []
        
        chunks = []
        total_messages = len(conversation)

        # sliding window logic
        step = self.window_size - self.window_overlap
        if step < 1: step = 1  # prevent dead loop

        for i in range(0, total_messages, step):
            # get the current window messages
            window_msgs = conversation[i : i + self.window_size]
            
            # 生成窗口唯一 ID (Episode ID), 例如: chat_001_w0, chat_001_w1
            window_index = i // step
            window_id = f"{conversation_id}_w{window_index}"
            
            # 格式化窗口数据 (核心逻辑)
            processed_data = self._format_window(window_msgs, window_id)
            
            chunk = {
                "id": window_id,                # 作为 KG 中的 Episode/Text 节点 ID
                "original_id": conversation_id, # 溯源到原始对话 ID
                "chunk_id": window_index,
                "text": processed_data['transcript'], # 纯文本抄本 (用于人类阅读或 Text Node 属性)
                "llm_content": processed_data['llm_content'], # 发给 API 的多模态 Payload
                "image_map": processed_data['image_map']      # 图片 ID 到 URL 的映射 (方便后续处理)
            }
            chunks.append(chunk)
            
            # 如果已经处理到最后一条消息，结束循环
            if i + self.window_size >= total_messages:
                break
                
        return chunks

    def _format_window(self, messages: List[Dict[str, Any]], window_id: str) -> Dict[str, Any]:
        """
        将消息列表转换为 OpenAI Multimodal Content 格式，并注入图片锚点。
        """
        llm_content = []
        transcript_lines = []
        image_map = {}
        img_local_idx = 0
        
        # # 添加引导语，帮助模型理解这是一个对话片段
        # llm_content.append(conversation_prefix)

        for msg in messages:
            role = msg.get('role', 'User')
            content_text = msg.get('content', '')
            images = msg.get('images', []) # 假设是 URL 列表
            
            # --- 1. 处理文本部分 ---
            # 构造: "**[User A]**: 车的保险杠坏了"
            display_text = f"\n**[{role}]**: {content_text}"
            
            llm_content.append({
                "type": "text",
                "text": display_text
            })
            
            # 更新纯文本抄本 (Transcript)
            transcript_lines.append(f"[{role}]: {content_text}")

            # --- 2. 处理图片部分 (关键逻辑) ---
            if images:
                for img_url in images:
                    # 生成全局唯一的图片 ID (作为 KG 节点 ID)
                    # 格式: IMG_{Window_ID}_{Index} -> IMG_chat001_w0_0
                    img_node_id = f"IMG_{window_id}_{img_local_idx}"
                    
                    # A. 注入文本锚点 (让 LLM 能“叫出”图片的名字)
                    # 注意：加换行符让锚点更明显
                    anchor_text = f"\n(Image uploaded by {role}: <{img_node_id}>)\n"
                    llm_content.append({
                        "type": "text",
                        "text": anchor_text
                    })
                    
                    # B. 插入真实图片对象
                    llm_content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": img_url,
                            "detail": "auto" # 可选 low/high/auto
                        }
                    })
                    
                    # 记录映射关系
                    transcript_lines.append(f"(Image: {img_node_id})")
                    image_map[img_node_id] = img_url
                    img_local_idx += 1

        return {
            "llm_content": llm_content,
            "transcript": "\n".join(transcript_lines),
            "image_map": image_map
        }

