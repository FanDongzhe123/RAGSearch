# run_naive_rag.py
import os
import json
import time
from tqdm import tqdm
from typing import List, Dict, Optional, Tuple
import argparse

from bing_search import (
    bing_web_search,
    extract_relevant_info,
    fetch_page_content,
    extract_snippet_with_context,
)
from evaluate import run_evaluation, extract_answer
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

import re
import string
from nltk.tokenize import sent_tokenize
import torch
from prompts import (
    get_task_instruction_openqa, 
    get_task_instruction_math, 
    get_task_instruction_multi_choice, 
    get_task_instruction_code, 
    get_naive_rag_instruction, 
)

def parse_args():
    parser = argparse.ArgumentParser(description="Run Naive RAG for various datasets and models.")

    # Dataset and split configuration
    parser.add_argument(
        '--dataset_name',
        type=str,
        required=True,
        choices=['gpqa', 'math500', 'aime', 'amc', 'livecode', 'nq', 'triviaqa', 'hotpotqa', '2wiki', 'musique', 'bamboogle', 'medmcqa', 'pubhealth'],
        help="Name of the dataset to use."
    )

    parser.add_argument(
        '--split',
        type=str,
        required=True,
        choices=['test', 'diamond', 'main', 'extended'],
        help="Dataset split to use."
    )

    parser.add_argument(
        '--subset_num',
        type=int,
        default=None,
        help="Number of examples to process. Defaults to all if not specified."
    )

    # Search and document retrieval configuration
    parser.add_argument(
        '--top_k',
        type=int,
        default=10,
        help="Number of top search results to retrieve."
    )

    parser.add_argument(
        '--max_doc_len',
        type=int,
        default=3000,
        help="Maximum length of each searched document."
    )

    # Model configuration
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help="Path to the pre-trained model."
    )

    parser.add_argument(
        '--use_jina',
        type=bool,
        default=True,
        help="Whether to use Jina API for document fetching."
    )

    parser.add_argument(
        '--jina_api_key',
        type=str,
        default='None',
        help="Your Jina API Key to Fetch URL Content."
    )

    # Sampling parameters
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.7,
        help="Sampling temperature."
    )

    parser.add_argument(
        '--top_p',
        type=float,
        default=0.8,
        help="Top-p sampling parameter."
    )

    parser.add_argument(
        '--top_k_sampling',
        type=int,
        default=20,
        help="Top-k sampling parameter."
    )

    parser.add_argument(
        '--repetition_penalty',
        type=float,
        default=None,
        help="Repetition penalty. If not set, defaults based on the model."
    )

    parser.add_argument(
        '--max_tokens',
        type=int,
        default=32768,
        help="Maximum number of tokens to generate. If not set, defaults based on the model and dataset."
    )

    # Bing API Configuration
    parser.add_argument(
        '--bing_subscription_key',
        type=str,
        required=True,
        help="Bing Search API subscription key."
    )

    parser.add_argument(
        '--bing_endpoint',
        type=str,
        default="https://api.bing.microsoft.com/v7.0/search",
        help="Bing Search API endpoint."
    )

    return parser.parse_args()

def main():
    args = parse_args()

    # Extract arguments
    dataset_name = args.dataset_name
    split = args.split
    subset_num = args.subset_num
    top_k = args.top_k
    max_doc_len = args.max_doc_len
    model_path = args.model_path
    temperature = args.temperature
    top_p = args.top_p
    top_k_sampling = args.top_k_sampling
    repetition_penalty = args.repetition_penalty
    max_tokens = args.max_tokens
    bing_subscription_key = args.bing_subscription_key
    bing_endpoint = args.bing_endpoint
    use_jina = args.use_jina
    jina_api_key = args.jina_api_key

    # Set default repetition_penalty if not provided
    if repetition_penalty is None:
        repetition_penalty = 1.05 if 'qwq' in model_path.lower() else 1.0
    
    if args.jina_api_key == 'None':
        jina_api_key = None

    # Paths to datasets
    if dataset_name == 'livecode':
        data_path = f'./data/LiveCodeBench/{split}.json'
    elif dataset_name in ['math500', 'gpqa', 'aime', 'amc']:
        data_path = f'./data/{dataset_name.upper()}/{split}.json'
    else:
        data_path = f'./data/QA_Datasets/{dataset_name}.json'

    # ---------------------- Caching Mechanism ----------------------
    # Define cache directories and file paths
    cache_dir = './cache'
    search_cache_path = os.path.join(cache_dir, 'search_cache.json')
    url_cache_path = os.path.join(cache_dir, 'url_cache.json')

    # Ensure cache directory exists
    os.makedirs(cache_dir, exist_ok=True)

    # Load existing caches or initialize empty dictionaries
    if os.path.exists(search_cache_path):
        with open(search_cache_path, 'r', encoding='utf-8') as f:
            search_cache = json.load(f)
    else:
        search_cache = {}

    if os.path.exists(url_cache_path):
        with open(url_cache_path, 'r', encoding='utf-8') as f:
            url_cache = json.load(f)
    else:
        url_cache = {}

    # Function to save caches
    def save_caches():
        with open(search_cache_path, 'w', encoding='utf-8') as f:
            json.dump(search_cache, f, ensure_ascii=False, indent=2)
        with open(url_cache_path, 'w', encoding='utf-8') as f:
            json.dump(url_cache, f, ensure_ascii=False, indent=2)

    # ---------------------- Model Loading ----------------------
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    # Define output directory based on model and dataset
    if 'qwq' in model_path.lower():
        if dataset_name in ['math500', 'gpqa', 'aime', 'amc', 'livecode']:
            output_dir = f'./outputs/{dataset_name}.qwq.naive_rag'
        else:
            output_dir = f'./outputs/runs.qa/{dataset_name}.qwq.naive_rag'
    else:
        model_short_name = model_path.split('/')[-1].lower().replace('-instruct', '')
        output_dir = f'./outputs/runs.baselines/{dataset_name}.{model_short_name}.naive_rag'
    os.makedirs(output_dir, exist_ok=True)

    # ---------------------- Data Loading ----------------------
    with open(data_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)
        if subset_num is not None:
            data = data[:subset_num]

    # ---------------------- Search and Document Retrieval ----------------------
    print("Performing Bing Web Searches for all questions...")

    # Initialize a list to hold relevant information for each question
    all_relevant_info = []

    for item in tqdm(data, desc="Searching"):
        question = item['Question']
        # Check if the question has already been searched and cached
        if question in search_cache:
            results = search_cache[question]
            # print(f"Using cached search results for question: {question}")
        else:
            if dataset_name == 'livecode':
                search_question = question[:500]
            else:
                search_question = question
            results = bing_web_search(search_question, bing_subscription_key, bing_endpoint, market='en-US', language='en')
            search_cache[question] = results
            # print(f"Executed and cached search for question: {question}")

        # Extract relevant information from search results
        relevant_info = extract_relevant_info(results)[:top_k]
        all_relevant_info.append(relevant_info)

    # Save search cache after retrieval
    save_caches()
    print("Search cache saved.")

    # Collect all unique URLs to fetch
    unique_urls = set()
    url_snippets_map = {}

    for relevant_info in all_relevant_info:
        for info in relevant_info:
            url = info['url']
            snippet = info.get('snippet', "")
            unique_urls.add(url)
            url_snippets_map[url] = snippet

    # Determine which URLs need to be fetched
    urls_to_fetch = [url for url in unique_urls if url not in url_cache]

    print(f"Fetching {len(urls_to_fetch)} unique URLs...")
    fetched_contents = fetch_page_content(
        urls_to_fetch,
        use_jina=use_jina,
        jina_api_key=jina_api_key,
        # snippets=url_snippets_map
    )

    # Update URL cache with fetched contents
    for url, content in fetched_contents.items():
        url_cache[url] = content

    # Save URL cache after fetching
    save_caches()
    print("URL cache saved.")

    # ---------------------- Prompt Construction ----------------------
    print("Constructing prompts for generation...")
    input_prompts = []

    for idx, item in enumerate(tqdm(data, desc="Constructing Prompts")):
        question = item['Question']

        formatted_documents = ""
        relevant_info = all_relevant_info[idx]
        for i, doc_info in enumerate(relevant_info):
            url = doc_info['url']
            snippet = doc_info.get('snippet', "")
            raw_context = url_cache.get(url, "")
            success, context = extract_snippet_with_context(raw_context, snippet, context_chars=max_doc_len)
            if success:
                context = context
            else:
                context = raw_context[:2 * max_doc_len]

            # Clean snippet from HTML tags if any
            clean_snippet = re.sub('<[^<]+?>', '', snippet)  # Removes HTML tags

            formatted_documents += f"**Document {i + 1}:**\n"
            formatted_documents += f"**Title:** {doc_info.get('title', '')}\n"
            formatted_documents += f"**URL:** {url}\n"
            formatted_documents += f"**Snippet:** {clean_snippet}\n"
            formatted_documents += f"**Content:** {context}\n\n"

        # Construct the instruction with documents and question
        instruction = get_naive_rag_instruction(question, formatted_documents)

        # Construct dataset and model-specific prompts
        if dataset_name in ['nq', 'triviaqa', 'hotpotqa', 'musique', 'bamboogle', '2wiki']:
            if 'qwq' in model_path.lower():
                user_prompt = get_task_instruction_openqa(question, model_name='qwq')
            else:
                user_prompt = get_task_instruction_openqa(question)

        elif dataset_name in ['math500', 'aime', 'amc']:
            if 'qwq' in model_path.lower():
                user_prompt = get_task_instruction_math(question, model_name='qwq')
            else:
                user_prompt = get_task_instruction_math(question)

        elif dataset_name == 'gpqa':
            if 'qwq' in model_path.lower():
                user_prompt = get_task_instruction_multi_choice(question, model_name='qwq')
            elif 'llama' in model_path.lower():
                user_prompt = get_task_instruction_multi_choice(question, model_name='llama')
            else:
                user_prompt = get_task_instruction_multi_choice(question)

        elif dataset_name == 'livecode':
            question_title = item.get('question_title', '')
            if 'qwq' in model_path.lower():
                user_prompt = get_task_instruction_code(question, question_title=question_title, model_name='qwq')
            else:
                user_prompt = get_task_instruction_code(question)
        else:
            user_prompt = ""  # Default to empty if dataset not matched

        # Combine instruction and user prompt
        full_prompt = instruction + "\n\n" + user_prompt

        # Apply tokenizer and chat template
        prompt = [{"role": "user", "content": full_prompt}]
        prompt = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        input_prompts.append(prompt)

    # ---------------------- Generation ----------------------
    # Initialize the LLM
    llm = LLM(
        model=model_path,
        tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=0.95,
    )

    print("Generating answers with LLM...")

    # Set default max_tokens if not provided
    if max_tokens is None:
        if 'qwq' in model_path.lower():
            max_tokens = 20480
        else:
            max_tokens = 10240

    start_time = time.time()
    # Generate model outputs
    output_list = llm.generate(
        input_prompts, 
        sampling_params=SamplingParams(
            max_tokens=max_tokens, 
            temperature=temperature, 
            top_p=top_p, 
            top_k=top_k_sampling, 
            repetition_penalty=repetition_penalty,
        )
    )

    total_time = time.time() - start_time

    # ---------------------- Evaluation ----------------------
    print("Evaluating generated answers...")
    run_evaluation(
        filtered_data=data,
        input_list=input_prompts,
        output_list=output_list,
        dataset_name=dataset_name,
        output_dir=output_dir,
        total_time=total_time,
        split=split,
    )

    # ---------------------- Update Search and URL Cache ----------------------
    print('Updating Search and URL Cache...')
    # Load existing caches or initialize empty dictionaries
    if os.path.exists(search_cache_path):
        with open(search_cache_path, 'r', encoding='utf-8') as f:
            search_cache_new = json.load(f)
    else:
        search_cache_new = {}

    if os.path.exists(url_cache_path):
        with open(url_cache_path, 'r', encoding='utf-8') as f:
            url_cache_new = json.load(f)
    else:
        url_cache_new = {}

    search_cache.update(search_cache_new)
    url_cache.update(url_cache_new)

    save_caches()

    print("Process completed.")

if __name__ == "__main__":
    main()
