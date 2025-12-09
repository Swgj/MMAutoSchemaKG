from neo4j import GraphDatabase
from tqdm import tqdm
from typing import Dict
import json
import logging


logger = logging.getLogger(__name__)


class MultimodalNeo4jLoader:
    """
    Load the multimodal extraction result(json-format) into neo4j database.
    """
    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str, database_name: str = "neo4j"):
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        self.database_name = database_name
        self.check_database_exist()
    
    def close(self):
        self.driver.close()

    def check_database_exist(self):
        with self.driver.session(database="system") as session:
            session.run(f"CREATE DATABASE `{self.database_name}` IF NOT EXISTS")
            return True
    
    def _check_apoc_availability(self):
        """Check if the database has installed the APOC plugin"""
        try:
            with self.driver.session(database=self.database_name) as session:
                session.run("RETURN apoc.version()").single()
        except Exception as e:
            logger.error("Error: APOC plugin not detected! This loader requires APOC.")
            logger.error(f"Details: {e}")
            raise e
        
    def clear_database(self):
        """WARNING: This will clear the entire database!"""
        with self.driver.session(database=self.database_name) as session:
            try:
                session.run("CALL apoc.periodic.iterate('MATCH (n) RETURN n', 'DETACH DELETE n', {batchSize:1000})")
            except:
                session.run("MATCH (n) DETACH DELETE n")
            logger.warning("Database cleared!")

    def create_constraints(self):
        """Create constraints for the database, accelerate the loading process"""
        queries = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (i:Image) REQUIRE i.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (ep:Episode) REQUIRE ep.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (ev:Event) REQUIRE ev.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (s:Session) REQUIRE s.id IS UNIQUE"
        ]
        with self.driver.session(database=self.database_name) as session:
            for q in queries:
                session.run(q)
        logger.info("Constraints created.")

    def load_extraction_result(self, extraction_result_path: str):
        """
        Load the multimodal extraction result into the database
        Support both json and jsonl
        """
        with open(extraction_result_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                f.seek(0)
                data = [json.loads(line) for line in f if line.strip()]
            logger.info(f"Loaded {len(data)} chunks from {extraction_result_path}")

            with self.driver.session(database=self.database_name) as session:
                for chunk in tqdm(data, desc="Importing chunks"):
                    self._import_single_chunk(session, chunk)
            logger.info("All chunks imported successfully!")

    
    def _import_single_chunk(self, session, chunk):
        """Import a single chunk into the database"""
        chunk_id = chunk.get('id')
        original_id = chunk.get('original_id')
        transcript = chunk.get('transcript')
        image_map = chunk.get('image_map', {})
        chunk_index = chunk.get('chunk_index')
        metadata = chunk.get('metadata', {})

        # 1. Create Episode Node(Chunk for Conversation)
        session.run("""
            // Create Session Node
            MERGE (s:Session {id: $original_id})
            SET s += $metadata

            // Create Episode Node(Chunk for Conversation)
            MERGE (ep:Episode {id: $id})
            SET ep += $metadata
            SET ep.transcript = $transcript,
                ep.original_session_id = $original_id,
                ep.chunk_index = $chunk_index

            // Hireachy: Session -> Episode
            MERGE (s)-[:HAS_CHUNK]->(ep)
        """, {'id': chunk_id, 'transcript': transcript, 'original_id': original_id, 'chunk_index': chunk_index, 'metadata': metadata})

        # 1.1 Create Chunk temperal relation Chunk -> next Chunk
        if chunk_index is not None and chunk_index > 0:
            prev_chunk_id = f"{original_id}_w{chunk_index - 1}"
            session.run("""
                MERGE (prev:Episode {id: $prev_id})
                MERGE (curr:Episode {id: $curr_id})
                MERGE (prev)-[:NEXT]->(curr)
            """, {'curr_id': chunk_id, 'prev_id': prev_chunk_id})
        
        # 2. Create Image Nodes(with URL)
        for img_id, img_url in image_map.items():
            session.run("""
                MERGE (i:Image {id: $id})
                SET i.url = $url
                SET i += $metadata
                MERGE (ep:Episode {id: $chunk_id})
                MERGE (ep)-[:CONTAINS_IMAGE]->(i)
            """, {'id': img_id, 'url': img_url, 'chunk_id': chunk_id, 'metadata': metadata})
        
        # 3. Import Entity-Relation Triplets
        er_list = chunk.get('entity_relation_dict', [])
        if er_list:
            for triple in er_list:
                head = triple.get('Head')
                relation = triple.get('Relation')
                tail = triple.get('Tail')
                
                if not all([head, relation, tail]): continue
                # Skip if the node is hallucinated image node
                if not all([self._is_valid_node(head, image_map), self._is_valid_node(tail, image_map)]): continue

                head_labels = ['Image'] if str(head).startswith('<IMG_') or str(head).startswith('IMG_') else ['Entity']
                tail_labels = ['Image'] if str(tail).startswith('<IMG_') or str(tail).startswith('IMG_') else ['Entity']
                
                head_clean = head.replace('<', '').replace('>', '')
                tail_clean = tail.replace('<', '').replace('>', '')

                relation_type = relation.upper().replace(' ', '_').replace('-', '_')

                query = """
                    CALL apoc.merge.node($h_labels, {id: $head_id}) YIELD node AS h
                    CALL apoc.merge.node($t_labels, {id: $tail_id}) YIELD node AS t
                    CALL apoc.merge.relationship(h, $r_type, {}, {}, t) YIELD rel
                    // Link back to Episode
                    WITH h, t
                    MATCH (ep:Episode {id: $chunk_id})
                    MERGE (ep)-[:MENTIONS]->(h)
                    MERGE (ep)-[:MENTIONS]->(t)
                """

                session.run(query, {
                    'h_labels': head_labels,
                    'head_id': head_clean,
                    't_labels': tail_labels,
                    'tail_id': tail_clean,
                    'r_type': relation_type,
                    'chunk_id': chunk_id
                })

        # 4. Import Event-Entity
        ee_list = chunk.get('event_entity_dict', [])
        if ee_list:
            for event in ee_list:
                event_text = event.get('Event')
                entities = event.get('Entity', [])

                if not event_text: continue

                # 4.1 Create Event Node, link to Episode
                session.run("""
                    MERGE (ev:Event {id: $event_id})
                    SET ev += $metadata
                    WITH ev
                    MATCH (ep:Episode {id: $chunk_id})
                    MERGE (ep)-[:CONTAINS_EVENT]->(ev)
                """, {'event_id': event_text, 'chunk_id': chunk_id, 'metadata': metadata})

                # 4.2 Link Event -> Entities (INVOLVES)
                for ent in entities:
                    ent_clean = ent.replace('<', '').replace('>', '')
                    ent_labels = ['Image'] if str(ent).startswith('<IMG_') or str(ent).startswith('IMG_') else ['Entity']

                    query_ee = """
                        MATCH (ev:Event {id: $event_id})
                        CALL apoc.merge.node($ent_labels, {id: $ent_id}) YIELD node AS e
                        MERGE (ev)-[:INVOLVES]->(e)
                    """
                    session.run(query_ee, {
                        'event_id': event_text,
                        'ent_labels': ent_labels,
                        'ent_id': ent_clean
                    })
                
        # 5. Import Event-Relation (Event -> Event)
        er_event_list = chunk.get('event_relation_dict', [])
        if er_event_list:
            for triple in er_event_list:
                head_evt = triple.get('Head')
                evt_relation = triple.get('Relation')
                tail_evt = triple.get('Tail')

                if not all([head_evt, evt_relation, tail_evt]): continue
                rel_type = evt_relation.upper().replace(' ', '_').replace('-', '_')

                query_ere = """
                    MERGE (h:Event {id: $head_id})
                    MERGE (t:Event {id: $tail_id})
                    WITH h, t
                    CALL apoc.merge.relationship(h, $r_type, {}, {}, t) YIELD rel

                    SET h += $metadata
                    SET t += $metadata

                    // make sure each event bind to the episode
                    WITH h, t, rel
                    MATCH (ep:Episode {id: $chunk_id})
                    MERGE (ep)-[:CONTAINS_EVENT]->(h)
                    MERGE (ep)-[:CONTAINS_EVENT]->(t)
                    
                    RETURN count(rel)
                """
                session.run(query_ere, {
                    'head_id': head_evt,
                    'tail_id': tail_evt,
                    'r_type': rel_type,
                    'chunk_id': chunk_id,
                    'metadata': metadata
                })

    def _is_valid_node(self, node_id: str, image_map: Dict[str, str]) -> bool:
        """Filter hallucinated image nodes"""
        cleaned_node_id = node_id.replace('<', '').replace('>', '')
        if str(cleaned_node_id).startswith('IMG_'):
            if cleaned_node_id not in image_map:
                logger.warning(f"Image node {node_id} not found in image map")
                return False
        return True