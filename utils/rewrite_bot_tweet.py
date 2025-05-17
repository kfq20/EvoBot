from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.distributed as dist
import pandas as pd
import torch

import json

import random
from tqdm import tqdm
import subprocess
import re
import os

def test_model_output(model_path, prompt, device):
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).to(device)
    print("=== successfully implemented llm! ===")
    tokenizer = AutoTokenizer.from_pretrained(model_path, model_max_length=2048)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model.config.pad_token_id = tokenizer.pad_token_id
    inputs = tokenizer(prompt, padding=True, truncation=True, return_tensors="pt", max_length=1024).to(device)
    output = model.generate(**inputs, max_new_tokens=1024, temperature=0.7, do_sample=True, repetition_penalty=1.3, pad_token_id = tokenizer.pad_token_id)
    output_text = tokenizer.batch_decode(output, skip_special_tokens=True)
    processed_outputs = []
    for output in output_text:
        response_output = re.sub(r'\[INST\].*?\[/INST\]', '', output, flags=re.DOTALL).strip()
        processed_outputs.append(response_output)
    print(f"=== prompt ===\n{prompt}\n")
    for output in processed_outputs:
        print(f"=== response ===\n{output}")
    

def rewrite_all_bot_tweets(gpu_id, model_path, bot_ids, community, queue):
    url_pattern1 = re.compile(r'https?://\S+')
    url_pattern2 = re.compile(r'http?://\S+')

    from .prompt_tempate import find_user_neighbors, llama_prompt
    torch.cuda.set_device(gpu_id)
    device = f"cuda:{gpu_id}"
    folder_path = f"./data/raw_data/community_{community}/"
    users_df = pd.read_json(folder_path + "user_summary.json")
    labels_df = pd.read_csv(folder_path + "label.csv")
    user_idx=labels_df['id']

    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).to(device)
    print("=== successfully implemented llm! ===")
    tokenizer = AutoTokenizer.from_pretrained(model_path, model_max_length=2048)
    model.config.pad_token_id = tokenizer.pad_token_id
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    batch_size = 4
    new_tweets = []
    for i in tqdm(range(int(len(bot_ids)//batch_size))):
        if i != int(len(bot_ids)//batch_size)-1:
            sampled_bot_ids = bot_ids[i*batch_size:(i+1)*batch_size]
        else:
            sampled_bot_ids = bot_ids[i*batch_size:]
        batch_prompt = []
        for bot_id in sampled_bot_ids:
            neighbor_ids = find_user_neighbors(folder_path + "edge.csv", bot_id, 5)
            neighbor_infos = []
            for neighbor_id in neighbor_ids:
                try:
                    neighbor_infos.append(users_df[users_df['id'] == neighbor_id].iloc[0])
                except:
                    pass
            user_info = users_df[users_df['id'] == bot_id].iloc[0]
            prompt = llama_prompt(user_info, neighbor_infos)
            batch_prompt.append(prompt)
        with torch.no_grad():
            inputs = tokenizer(batch_prompt, padding=True, truncation=True, return_tensors="pt", max_length=2048).to(device)
            output = model.generate(**inputs, max_new_tokens=512, temperature=0.7, do_sample=True, repetition_penalty=1.3, pad_token_id = tokenizer.pad_token_id)
            output_text = tokenizer.batch_decode(output, skip_special_tokens=True)
            processed_outputs = []
            for output in output_text:
                response_output = re.sub(r'\[INST\].*?\[/INST\]', '', output, flags=re.DOTALL).strip()
                processed_outputs.append(response_output)
        
        for bot_id, text in zip(sampled_bot_ids, processed_outputs):
            if re.search(r'\n', text):
                segments = re.split(r'\n', text.strip())
                segments = [segments[i].strip() for i in range(1, len(segments))]
                results = []
                for item in segments:
                    item = re.sub(r"Tweet", "", item).strip()
                    item = re.sub(r"Example", "", item).strip()
                    item = re.sub(r'^[^\w]*', '', item).strip()
                    item = re.sub(url_pattern1, '', item).strip()
                    item = re.sub(url_pattern2, '', item).strip()
                    if re.match(r"^\d", item):
                        item = re.sub(r"^\d+\s*[^\w\s]*\s*", "", item).strip()
                    # else:
                    #     item = re.sub(r"^.*?(\n+)", "", item, count=1).strip()
                    results.append(item)
            else:
                results = [text]
            new_tweets.append({
                'author_id': bot_id,
                'text':results
            })
    print(f"=== Process {gpu_id} done ===") 
    queue.put((gpu_id, new_tweets))

import os
import subprocess

def get_free_gpus():
    """
    Get a list of free GPUs in the current environment (restricted by CUDA_VISIBLE_DEVICES), returning logical GPU indices.
    """
    try:
        # Get the CUDA_VISIBLE_DEVICES environment variable
        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cuda_visible_devices is not None:
            visible_gpus = list(map(int, cuda_visible_devices.split(',')))
        else:
            result = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                encoding="utf-8",
            )
            visible_gpus = list(range(len(result.strip().split("\n"))))

        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            encoding="utf-8",
        )
        memory_used = [int(x) for x in result.strip().split("\n")]

        free_physical_gpus = [idx for idx in visible_gpus if memory_used[idx] < 100]

        free_logical_gpus = [visible_gpus.index(phys_gpu) for phys_gpu in free_physical_gpus]

        return free_logical_gpus
    except Exception as e:
        print(f"Error while checking GPUs: {e}")
        return []

    
if __name__ == '__main__':
    test_model_output(
        model_path="./models/SFT/sft_merged_ckp",
                    prompt="hello",
                      device="cuda:0")