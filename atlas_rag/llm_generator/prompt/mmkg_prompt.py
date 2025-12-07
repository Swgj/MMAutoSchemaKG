conversation_prefix = {
    "type": "text", 
    "text": "Below is a conversation segment containing text and images. "
            "Images are explicitly tagged with IDs (e.g., <IMG_...>). "
            "Please refer to these IDs when extracting entities or relations involving visual evidence.\n" 
            + "-" * 20 + "\n"
}

MULTIMODAL_TRIPLE_INSTRUCTIONS = {
    "en": {
        "system": "You are a multimodal knowledge graph expert who extracts structured information from conversations containing text and images.",
        
        "entity_relation": """Given a conversation segment, summarize all important entities and relations.
        
        **CRITICAL INSTRUCTIONS FOR IMAGES:**
        1. You will see tags like '<IMG_session_w0_1>'. These represent **Image Entities**.
        2. Always extract these Image IDs as entities when they are relevant.
        3. Extract relations that describe:
           - Who sent the image? (e.g., 'User' -> 'uploaded' -> '<IMG_...>')
           - What does the image depict? (e.g., '<IMG_...>' -> 'depicts' -> 'Damage')
           - How does the text relate to the image? (e.g., 'Scratch' -> 'visible_in' -> '<IMG_...>')
        
        Output valid JSON strictly following this schema:
        [
            {"Head": "{noun or IMG_ID}", "Relation": "{verb}", "Tail": "{noun or IMG_ID}"},
            ...
        ]
        """,
        
        "event_entity": """Identify events and the entities (including images) involved in them.
        If an image is proof of an event, include the Image ID in the entity list.
        Output format:
        [
            {"Event": "{sentence}", "Entity": ["{noun}", "{IMG_ID}", ...]},
            ...
        ]
        """,
        
        "event_relation": """Identify temporal and causal relationships between events.
        Output format:
        [
            {"Head": "{Event 1}", "Relation": "{relation}", "Tail": "{Event 2}"},
            ...
        ]
        """
    }
}

IMAGE_CAPTION_PROMPT = """
Describe the main visual content of this image concisely for a knowledge graph retrieval system. 
Include key objects, visible text (if any), and the overall context.
"""