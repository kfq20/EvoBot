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
from torch.multiprocessing import Process, Queue, set_start_method
import random
from tqdm import tqdm
import os
os.environ["WANDB_DISABLED"]="true"

PRECISION_MAPPING = {16: torch.float16, 32: torch.float32, 64: torch.float64}

generator_device = 'cuda:1'
other_device = "cuda:2"

with open("/home/fanqi/llm_simulation/config.json", "r") as config_file:
    config = json.load(config_file)

COMM = 2

tweets_df = pd.read_json(f"/home/fanqi/llm_simulation/data/raw_data/community_{COMM}/tweet.json")
users_df = pd.read_json(f"/home/fanqi/llm_simulation/data/raw_data/community_{COMM}/user_info.json")
labels_df = pd.read_csv(f"/home/fanqi/llm_simulation/data/raw_data/community_{COMM}/label.csv")
user_idx=labels_df['id']
uid_index={uid:index for index,uid in enumerate(user_idx.values)}
bot_ids = labels_df[labels_df['label'] == 'bot']['id'].tolist()

processed_data_folder='/home/fanqi/llm_simulation/data/processed_data/'
dataset=Twibot22(root=processed_data_folder,device=other_device,process=False,save=False)
feature_extractor = Feature_extractor(model=config["feature_extractor"]["model"])
des_tensor,tweets_tensor,num_prop,category_prop,edge_index,edge_type,labels,train_idx,val_idx,test_idx=dataset.dataloader()

model_path = "/home/fanqi/llm_simulation/models/SFT/sft_merged_ckp"
folder_path = "/home/fanqi/llm_simulation/data/raw_data"
dpo_dataset_path = "/home/fanqi/llm_simulation/data/dpo_data"
model_discriminator = BotRGCN(cat_prop_size=config["discriminator"]["cat_prop_size"],
                              embedding_dimension=config['discriminator']["embedding_dimension"]).to(other_device)
optimizer_discriminator = torch.optim.AdamW(model_discriminator.parameters(),
                        lr=config["discriminator"]["lr"],weight_decay=config["discriminator"]["weight_decay"])
loss_func_discriminator = nn.CrossEntropyLoss()

DPO_config = DPOConfig(**config["DPO_trainer"], bf16=True, gradient_checkpointing=True)
peft_config = LoraConfig(**config["peft"])

dpo_dataset_size = config["DPO_dataset_size"]
batch_size = config["generator_training"]["batch_size"]
training_epoch = config["generator_training"]["epoch"]

set_start_method('spawn', force=True) 
num_gpus = torch.cuda.device_count()
bot_ids_per_gpu = len(bot_ids) // num_gpus 
queue = Queue()
processes = []
for gpu_id in range(num_gpus):
    start_idx = gpu_id * bot_ids_per_gpu
    end_idx = start_idx + bot_ids_per_gpu
    bot_ids_gpu = bot_ids[start_idx:end_idx]
    p = Process(target=rewrite_all_bot_tweets, args=(gpu_id, model_path, bot_ids_gpu, COMM, queue))
for p in processes:
    p.join()

all_new_tweets = []
while not queue.empty():
    all_new_tweets.extend(queue.get())
with open(f'/home/fanqi/llm_simulation/data/raw_data/{COMM}/dpo_{epoch}_bot_tweet.json', 'w') as f:
    json.dump(all_new_tweets, f, ensure_ascii=False, indent=4)