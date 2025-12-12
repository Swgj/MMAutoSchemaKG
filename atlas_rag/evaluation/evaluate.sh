#!/bin/bash
PYTHON_EXEC=/opt/homebrew/anaconda3/envs/mmauto/bin/python

# evaluate mmrag on locomo hard
# for i in {0..9}
# do
#     echo "Evaluating locomo-hard-$i..."
#     $PYTHON_EXEC atlas_rag/evaluation/mm_locomo_evaluation.py \
#         --neo4j_database "locomo-hard-$i" \
#         --data_path "example_data/locomo_hard_qa/locomo_hard_qa_$i.json" \
#         --output_path "generation_result/gemini-2.5-flash/mmrag_evaluation/locomo-hard-$i-qa-hipporag-node.json" \
#         --model_name "gemini-2.5-flash" \
#         --embedding_model_name "gemini-embedding-001" \
#         --max_workers 5
# done

# benchmark
# for i in {1..9}
# do
#     echo "Evaluating locomo-hard-$i benchmark mode..."
#     $PYTHON_EXEC atlas_rag/evaluation/mm_locomo_evaluation.py \
#         --neo4j_database "locomo-hard-$i" \
#         --data_path "example_data/locomo_hard_qa/locomo_hard_qa_$i.json" \
#         --output_path "generation_result/gemini-2.5-flash/mmrag_evaluation/locomo-hard-$i-qa-benchmark.json" \
#         --model_name "gemini-2.5-flash" \
#         --embedding_model_name "gemini-embedding-001" \
#         --max_workers 5 \
#         --benchmark_mode
# done

# compute recall for hipporag
for i in {0..9}
do
    echo "Evaluating locomo-hard-$i..."
    $PYTHON_EXEC atlas_rag/evaluation/mm_locomo_evaluation.py \
        --neo4j_database "locomo-hard-$i" \
        --data_path "example_data/locomo_hard_qa/locomo_hard_qa_$i.json" \
        --output_path "generation_result/gemini-2.5-flash/mmrag_evaluation/locomo-hard-$i-qa-hipporag-node_recall.json" \
        --model_name "gemini-2.5-flash" \
        --embedding_model_name "gemini-embedding-001" \
        --max_workers 5 \
        --recall_compute
done
