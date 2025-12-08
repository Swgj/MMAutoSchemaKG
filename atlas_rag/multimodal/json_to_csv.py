import os
import csv
import json
from tqdm import tqdm

class MultimodalJsonToCsv:
    """
    Convert Multimodal Extraction JSON results to CSVs.
    Compatible with the original Atlas RAG data format but supports Image nodes.
    """
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Output files
        self.node_file = os.path.join(output_dir, "entity_nodes.csv")
        self.edge_file = os.path.join(output_dir, "entity_edges.csv")
        self.text_node_file = os.path.join(output_dir, "text_nodes.csv")
        self.text_edge_file = os.path.join(output_dir, "text_edges.csv") 
        self.image_map_file = os.path.join(output_dir, "image_map.json")

    def convert(self, json_file_path):
        if not os.path.exists(json_file_path):
            print(f"❌ File not found: {json_file_path}")
            return

        print(f"🔄 Converting {json_file_path} to CSVs...")
        
        visited_nodes = set()
        visited_texts = set()
        image_url_map = {}

        # Load data
        with open(json_file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                f.seek(0)
                data = [json.loads(line) for line in f if line.strip()]

        # Open all CSV writers
        with open(self.text_node_file, "w", newline='', encoding='utf-8') as f_tn, \
             open(self.text_edge_file, "w", newline='', encoding='utf-8') as f_te, \
             open(self.node_file, "w", newline='', encoding='utf-8') as f_n, \
             open(self.edge_file, "w", newline='', encoding='utf-8') as f_e:

            # Headers (Aligned with original project schema)
            writer_tn = csv.writer(f_tn)
            writer_tn.writerow(["text_id:ID", "original_text", ":LABEL"])
            
            writer_te = csv.writer(f_te)
            writer_te.writerow([":START_ID", ":END_ID", ":TYPE"])
            
            writer_n = csv.writer(f_n)
            writer_n.writerow(["name:ID", "type", ":LABEL"])
            
            writer_e = csv.writer(f_e)
            writer_e.writerow([":START_ID", ":END_ID", "relation", ":TYPE"])

            for chunk in tqdm(data, desc="Processing Chunks"):
                # Use chunk_id (e.g., session_1_w0) as the Text ID
                chunk_id = chunk['id']
                transcript = chunk.get('transcript', '').replace("\n", " ")
                
                # 1. Save Image Map
                if 'image_map' in chunk:
                    image_url_map.update(chunk['image_map'])

                # 2. Write Text Node (Passage)
                if chunk_id not in visited_texts:
                    writer_tn.writerow([chunk_id, transcript, "Passage"])
                    visited_texts.add(chunk_id)

                # 3. Collect all triples (Entity-Relation + Event-Relation)
                all_triples = chunk.get('entity_relation_dict', []) + chunk.get('event_relation_dict', [])
                
                # 4. Collect Event-Entity pairs (convert to triples for CSV)
                ee_list = chunk.get('event_entity_dict', [])
                for item in ee_list:
                    event = item.get('Event')
                    entities = item.get('Entity', [])
                    if event and entities:
                        for ent in entities:
                            all_triples.append({"Head": event, "Relation": "INVOLVES", "Tail": ent})

                # 5. Process all triples
                for triple in all_triples:
                    head = triple.get('Head')
                    relation = triple.get('Relation')
                    tail = triple.get('Tail')
                    
                    if not all([head, relation, tail]): continue
                    
                    # Clean IDs
                    head = head.replace('<', '').replace('>', '')
                    tail = tail.replace('<', '').replace('>', '')

                    # --- Write Nodes ---
                    for node_id in [head, tail]:
                        if node_id not in visited_nodes:
                            # Identify type: Image or Entity
                            node_type = "Image" if str(node_id).startswith("IMG_") else "Entity"
                            if node_id == head and "Event" in triple: node_type = "Event" # Simple heuristic
                            
                            writer_n.writerow([node_id, node_type, "Node"])
                            visited_nodes.add(node_id)
                            
                            # --- Write Edge: Entity -> Text (Provenance) ---
                            # Required by HippoRAG to map back to text
                            writer_te.writerow([node_id, chunk_id, "APPEARS_IN"])

                    # --- Write Edge: Entity -> Entity ---
                    writer_e.writerow([head, tail, relation, "RELATION"])

        # Save Image Map
        with open(self.image_map_file, 'w', encoding='utf-8') as f:
            json.dump(image_url_map, f, indent=2, ensure_ascii=False)
            
        print(f"✅ CSV conversion complete! Output dir: {self.output_dir}")

if __name__ == "__main__":
    # Example Usage
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True, help="Input JSON file path")
    parser.add_argument("--out", required=True, help="Output CSV directory")
    args = parser.parse_args()
    
    converter = MultimodalJsonToCsv(args.out)
    converter.convert(args.json)