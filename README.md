# RAGSearch
## Installation
### Retriever Backends
We recommend creating a separate conda environment for each retriever backend.
#### Dense-RAG
```bash
conda create -p ./env/retriever python=3.10
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install transformers datasets pyserini
conda install -c pytorch -c nvidia faiss-gpu=1.8.0
pip install uvicorn fastapi
```
