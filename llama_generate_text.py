from datasets import load_dataset
from trl import DPOConfig, DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.distributed as dist
import pandas as pd
import torch
from torch import nn
from peft import (
    LoraConfig,
    get_peft_model,
)
from peft import PeftModel
import json
from models.discriminator import BotRGCN, train_discrim, test_discrim
from models.generator import Generator
from Dataset import Twibot22
from models.feature_extrator import Feature_extractor
from utils.prompt_tempate import find_user_neighbors, llama_prompt
from utils.merge_peft_adapters import merge_peft_adapters
from utils.rewrite_bot_tweet import rewrite_all_bot_tweets
import torch.multiprocessing as mp
import random
from tqdm import tqdm
import os
import re
os.environ["WANDB_DISABLED"]="true"

if __name__ == '__main__':
    COMM = 2
    labels_df = pd.read_csv(f"/home/fanqi/llm_simulation/data/raw_data/community_{COMM}/label.csv")
    human_ids = labels_df[labels_df['label'] == 'human']['id'].tolist()
    bot_ids = random.sample(human_ids, 500)
    model_path = "NousResearch/Llama-2-7b-chat-hf"
    user_idx=labels_df['id']
    mp.set_start_method('spawn', force=True) 
    num_gpus = torch.cuda.device_count()
    bot_ids_per_gpu = len(bot_ids) // num_gpus 
    manager = mp.Manager()
    queue = manager.Queue()
    processes = []
    for gpu_id in range(num_gpus):
        start_idx = gpu_id * bot_ids_per_gpu
        end_idx = start_idx + bot_ids_per_gpu
        if gpu_id != num_gpus - 1:
            bot_ids_gpu = bot_ids[start_idx:end_idx]
        else:
            bot_ids_gpu = bot_ids[start_idx:]
        # rewrite_all_bot_tweets(gpu_id, model_path, bot_ids_gpu, COMM)
        p = mp.Process(target=rewrite_all_bot_tweets, args=(gpu_id, model_path, bot_ids_gpu, COMM, queue))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    all_new_tweets = []
    while not queue.empty():
        all_new_tweets.extend(queue.get())

    with open(f'/home/fanqi/llm_simulation/data/raw_data/community_{COMM}/llama_human_tweet.json', 'w') as f:
        json.dump(all_new_tweets, f, ensure_ascii=False, indent=4)