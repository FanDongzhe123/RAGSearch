export OPENAI_API_KEY="sk-9533fa58a84848f0ac6d9ea869ef9eed"
export OPENAI_BASE_URL="https://api.deepseek.com/v1"


SPACY_MODEL="en_core_web_trf"
EMBEDDING_MODEL="sentence-transformers/all-mpnet-base-v2"
DATASET_NAME="nq_5000"
LLM_MODEL="deepseek-chat"
MAX_WORKERS=16

python run.py \
    --spacy_model ${SPACY_MODEL} \
    --embedding_model ${EMBEDDING_MODEL} \
    --dataset_name ${DATASET_NAME} \
    --llm_model ${LLM_MODEL} \
    --max_workers ${MAX_WORKERS}