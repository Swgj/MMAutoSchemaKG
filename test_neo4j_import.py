import os
import argparse
import logging
from atlas_rag.multimodal.neo4j_loader import MultimodalNeo4jLoader

logger = logging.getLogger(__name__)

def main(args=None):
    if not args:
        parser = argparse.ArgumentParser(description="Import Multimodal KG Extraction Results to Neo4j")
        
        # 默认值适配你的项目环境
        parser.add_argument("--file", type=str, required=True, 
                            help="Path to the extraction result JSON file")
        parser.add_argument("--uri", type=str, default="bolt://localhost:7687", 
                            help="Neo4j URI (default: bolt://localhost:7687)")
        parser.add_argument("--user", type=str, default="neo4j", 
                            help="Neo4j User (default: neo4j)")
        parser.add_argument("--password", type=str, default="password", 
                            help="Neo4j Password (default: password)")
        parser.add_argument("--clear", action="store_true", 
                            help="[DANGER] Clear the database before importing")

        args = parser.parse_args()
    logger.info(f"Connecting to Neo4j at {args.uri}...")

    try:
        neo4j_loader = MultimodalNeo4jLoader(args.uri, args.user, args.password)

        if args.clear:
            logger.warning("Clearing the database before importing...")
            neo4j_loader.clear_database()

        logger.info("Creating constraints for the database...")
        neo4j_loader.create_constraints()

        logger.info(f"Importing data from {args.file}...")
        neo4j_loader.load_extraction_result(args.file)

        logger.info("All done!")
    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        if 'neo4j_loader' in locals():
            neo4j_loader.close()
            logger.info("Connection closed")

if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    
    file_path = 'generation_result/gemini-2.5-flash/kg_extraction/gemini-2.5-flash_locomo_hard_0_output_20251207005357_1_in_1.json'
    args = argparse.Namespace(
        file=file_path,
        uri="bolt://localhost:7687",
        user="neo4j",
        password="password",
        clear=True
    )
    main(args)