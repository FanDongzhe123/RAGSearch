import os
import json
import logging
import shutil
from tqdm import tqdm
from src.hipporag import HippoRAG

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

if os.path.exists("openai_api_key.txt"):
    os.environ["OPENAI_API_KEY"] = open("openai_api_key.txt").read().strip()

def stream_load_jsonl(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                doc = json.loads(line)
                text = doc.get("contents", "").strip()
                if text:
                    yield text
            except Exception as e:
                logger.warning(f"Error reading line: {e}")
                continue

def build_and_copy_insert(dataset, base_path, embedding_model, llm_model):
    stages = [
        # ("corpus_train_1000_w_test.jsonl", "hipporag_1000"),
        # ("corpus_train_increment_3000.jsonl", "hipporag_3000"),
        ("corpus_train_increment_5000.jsonl", "hipporag_5000"),
    ]

    prev_dir = None
    for idx, (filename, model_tag) in enumerate(stages):
        input_path = os.path.join(base_path, filename)
        output_dir = os.path.join("./Graphrags", dataset, model_tag)
        os.makedirs(output_dir, exist_ok=True)

        if idx == 0:
            # ① 第一阶段：从头构建
            logger.info(f"🚀 Building from scratch: {output_dir}")
            hipporag = HippoRAG(
                save_dir=output_dir,
                llm_model_name=llm_model,
                embedding_model_name=embedding_model
            )
        else:
            # ② 后续阶段：复制上一个目录
            logger.info(f"🔁 Copying {prev_dir} → {output_dir}")
            shutil.copytree(prev_dir, output_dir, dirs_exist_ok=True)
            hipporag = HippoRAG(
                save_dir=output_dir,
                llm_model_name=llm_model,
                embedding_model_name=embedding_model
            )

        # ③ 插入数据
        logger.info(f"📘 Reading from: {input_path}")
        docs = list(stream_load_jsonl(input_path))
        with tqdm(total=len(docs), desc=f"Indexing {model_tag}", unit="docs") as pbar:
            for i in range(0, len(docs), 8):
                batch = docs[i:i + 8]
                hipporag.index(docs=batch)
                pbar.update(len(batch))

        logger.info(f"✅ Completed {model_tag} ({len(docs)} docs)")
        prev_dir = output_dir  # 下一阶段复制的来源目录

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--embed", type=str, default="nvidia/NV-Embed-v2")
    parser.add_argument("--llm", type=str, default="gpt-4o-mini")
    args = parser.parse_args()

    dataset = args.dataset
    base_path = f"./datasets/{dataset}"
    build_and_copy_insert(dataset, base_path, args.embed, args.llm)
