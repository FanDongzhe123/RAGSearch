LLM_API_KEY = "EMPTY"
MODEL_NAME = "Qwen2.5-7B-Instruct"
HUGGINGFACE_MODEL_NAME = "Qwen2.5-7B-Instruct"
EMBED_MODEL_NAME = "jinaai/jina-embeddings-v3"
GRAG_MODE = {"lightrag": "hybrid", "minirag": "light", "hypergraphrag": "hybrid", "pathrag": "hybrid"}
RAG_SERVER_CONFIG = {
    "hypergraphrag": "http://127.0.0.1:8336/search",
    "hipporag2": "http://127.0.0.1:8316/search",
    "linearrag": "http://127.0.0.1:8356/search",
    "raptor": "http://127.0.0.1:8346/search",
    "graphrag": "http://127.0.0.1:8326/search",
    "dense": "http://127.0.0.1:8306/retrieve"
}
# Naive RAG: remote retriever (Search-R1 style). If set, naive_rag_reasoning uses it instead of local vdb_retrieve.
NAIVE_RAG_RETRIEVER_URL = "http://127.0.0.1:8205/retrieve"  # Search-R1 style; override via env if needed
