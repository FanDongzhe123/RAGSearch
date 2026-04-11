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
cd LinearRAG
pip install -r requirements.txt
# Download Spacy language model
python -m spacy download en_core_web_trf
```

#### RAPTOR
```bash
conda create -p ./env/raptor python=3.9
conda activate ./env/raptor
cd GraphR1
cd raptor
pip install -r requirements.txt
```

#### GraphRAG
```bash
conda create -p ./env/graphrag python=3.10
conda activate ./env/graphrag
python -m pip install graphrag
```

#### HyperGraphRAG
HypergraphRAG shares the same environment as Graph-R1
```bash
conda activate ./env/graph-r1
```
### Training-free agentic systems
#### Search-o1
```bash
conda create -p ./env/search_o1 python=3.9
conda activate ./env/search_o1
cd Search-o1
pip install -r requirements.txt
```

#### GraphSearch
```bash
conda create -p ./env/graphsearch python=3.11
conda activate ./env/graphsearch
cd GraphSearch
pip install -r requirements.txt
```

### RL-based agentic systems
#### Search-R1
```bash
conda create -p ./env/searchr1 python=3.9
conda activate ./env/searchr1
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip3 install vllm==0.6.3
pip install -e .
pip3 install flash-attn --no-build-isolation
pip install wandb
```

#### Graph-R1
```bash
conda create -p ./env/graphr1 python==3.11.11
conda activate ./env/graphr1
pip3 install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip3 install flash-attn --no-build-isolation
pip3 install -e .
cd Graph-R1
pip3 install -r requirements.txt
```

## Offline GraphRAG Construction
This is an example of constructing GraphRAG on HippoRAG2; other retrieve backends follow a similar process.
- Step 1 Set your openai API key in `openai_api_key_txt`
- Step 2 Build the GraphRAG with `build_hipporag_increment.py`

## Quick Start
### Graph-R1
```bash
cd GraphR1
# -a is the retriever server port number; -d is the training dataset -p is the backbone path; -m is the backbone name; -r is the retriever backend; -b is the batch size
bash run_graphr1.sh -a 8007 -d NQ-HotpotQA -p Qwen/Qwen2.5-7B-Instruct -m Qwen2.5-7B-Instruct -r hipporag2 -b 128 -g 4
```



