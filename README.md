# RAGSearch
## Installation
### Retriever Backends
We recommend creating a separate conda environment for each retriever backend.
#### Dense-RAG
```bash
conda create -p ./env/retriever python=3.10
conda activate ./env/retriever
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install transformers datasets pyserini
conda install -c pytorch -c nvidia faiss-gpu=1.8.0
pip install uvicorn fastapi
```

#### HippoRAG2
```bash
conda create -p ./env/hipporag python=3.10
conda activate ./env/hipporag
pip install hipporag
```

#### LinearRAG
```bash
conda create -p ./env/linearrag python=3.9
conda activate ./env/linearrag
cd Graph-R1
cd Linearrag
```
