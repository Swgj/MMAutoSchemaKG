#!/bin/bash
PYTHON_EXEC=/opt/homebrew/anaconda3/envs/mmauto/bin/python
BASE_DIR="generation_result/gemini-2.5-flash/kg_extraction"


# 1. extract kg from original json file
# which is done in test_mmkg.sh

# 2. import kg to neo4j
# with test_neo4j_import.py

# 3. build embedding for each kg
# with altas_rag.multimodal.vector_store.py

# for i in {0..9}
do
    # Step 1 finished, skip

    # Step 2
    matches=( "${BASE_DIR}"/*hard_${i}_*.json )
    target_file="${matches[0]}"
    if [ ! -f "$target_file" ]; then
        echo "Warning: File for locomo_hard_$i not found, skipping..."
        continue
    fi

    database_name="locomo-hard-$i"
    echo "Importing kg to neo4j for $database_name..."
    echo "File: ${target_file}"
    echo "Database Name: $database_name"
    # echo "Clear: False"
    echo "Clear: True"
    sleep 5

    $PYTHON_EXEC example_scripts/test_neo4j_import.py --file "$target_file" --database_name "$database_name" --clear
    # $PYTHON_EXEC example_scripts/test_neo4j_import.py --file "$target_file" --database_name "$database_name"
    
    echo "Import Done"
    echo "--------------------------------"
    sleep 5


    # Step 3
    echo "Building embedding for $database_name..."
    $PYTHON_EXEC atlas_rag/multimodal/vector_store.py --database_name "$database_name"
    echo "Embedding Done"
    echo "--------------------------------"
    sleep 5
done