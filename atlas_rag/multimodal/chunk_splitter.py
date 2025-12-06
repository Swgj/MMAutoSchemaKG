"""
Conversation Chunk Splitter
"""
from typing import List, Dict, Any
from atlas_rag.multimodal.datamodel import ConversationSession, ProcessedChunk, MultimodalMessage

class ConversationSplitter:
    """
    Split a long conversation into windows (Chunks) and orchestrate the formatting
    using the logic encapsulated in MultimodalMessage.
    """
    
    def __init__(self, window_size: int = 10, window_overlap: int = 2):
        self.window_size = window_size
        self.window_overlap = window_overlap

    def split_messages(self, session: ConversationSession) -> List[ProcessedChunk]:
        """
        Split session messages into processed chunks ready for LLM extraction.
        """
        chunks = []
        messages = session.messages
        total_msgs = len(messages)
        
        msg_img_counts = [0] * total_msgs # use msg_img_counts[i] represent img count before i-th msg
        cur_img_idx = 0
        for i in range(total_msgs):
            msg_img_counts[i] = cur_img_idx
            cur_img_idx += len(messages[i].img_url)
        
        # 1. Calculate step size
        step = self.window_size - self.window_overlap
        if step < 1: 
            step = 1

        # 2. Sliding window loop
        for i in range(0, total_msgs, step):
            # Slice current window
            window_msgs = messages[i : i + self.window_size]
            
            # Generate unique Window ID (e.g., "session_123_w0")
            window_idx = i // step
            chunk_id = f"{session.id}_w{window_idx}"

            global_img_start_idx = msg_img_counts[i]
            
            # 3. Process the window by delegating to DataModel methods
            processed_data = self._process_window(window_msgs, session.id, global_img_start_idx)
            
            # 4. Create the final Chunk object
            chunk = ProcessedChunk(
                chunk_id=chunk_id,
                original_id=session.id,
                window_index=window_idx,
                transcript=processed_data['transcript'],
                llm_payload=processed_data['llm_payload'],
                image_map=processed_data['image_map'],
                session_metadata=session.metadata
            )
            chunks.append(chunk)
            
            # Stop if we've covered the last message
            if i + self.window_size >= total_msgs:
                break
                
        return chunks

    def _process_window(self, messages: List[MultimodalMessage], session_id: str, global_img_start_idx: int) -> Dict[str, Any]:
        """
        Orchestrates the formatting of a list of messages.
        It calls `format_to_openai_content` on each message instance.
        """
        llm_payload = []
        full_image_map = {}
        transcript_lines = []
        
        # Global counter for images within this specific chunk
        current_img_idx = global_img_start_idx
        
        # Add System-level instruction at the start of the User content list
        llm_payload.append({
            "type": "text",
            "text": "Analyze the following conversation segment. "
                    "Note that images are tagged with IDs (e.g., <IMG_...>). "
                    "Treat these IDs as entities when extracting relations.\n" 
                    + "-" * 20 + "\n"
        })

        for msg in messages:
            # --- 1. Call the method defined in your DataModel ---
            # This handles formatting, ID generation, and anchor injection internally
            result = msg.format_to_openai_content(
                img_start_index=current_img_idx,
                id_prefix=session_id
            )
            
            # --- 2. Aggregate results ---
            llm_payload.extend(result["content_list"])
            full_image_map.update(result["image_map"])
            transcript_lines.append(result["transcript"])
            # Update the index counter so the next message gets the correct sequence number
            current_img_idx = result["next_img_idx"]

        return {
            "llm_payload": llm_payload,
            "image_map": full_image_map,
            "transcript": "\n".join(transcript_lines)
        }