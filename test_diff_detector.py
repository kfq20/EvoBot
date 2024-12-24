from datasets import load_dataset
from trl import DPOConfig, DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch.distributed as dist
import pandas as pd
import torch
from torch import nn
import numpy as np
from peft import (
    LoraConfig,
    get_peft_model,
)
from peft import PeftModel
import json
from models.discriminator import BotRGCN, train_discrim, test_discrim
from models.generator import Generator
from Dataset import Twibot22
from data.raw_data.preprocess import extract_tweets_1, tweets_embedding
from models.feature_extrator import Feature_extractor
from utils.prompt_tempate import find_user_neighbors, llama_prompt
from utils.merge_peft_adapters import merge_peft_adapters
from utils.rewrite_bot_tweet import rewrite_all_bot_tweets, get_free_gpus
from torch.multiprocessing import Process, Queue, set_start_method, Manager
import torch.nn.functional as F
import random
from tqdm import tqdm
import os
import re
SEED = 3401
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)
# transformers.set_seed(42)

os.environ["WANDB_DISABLED"]="true"
# os.environ["WANDB_DISABLED"]="true"

PRECISION_MAPPING = {16: torch.float16, 32: torch.float32, 64: torch.float64}
COMM = 5
generator_version = "SFT"
processed_data_folder=f"/home/fanqi/llm_simulation/data/processed_data/community_{COMM}/"
bot_processed_data = f"/home/fanqi/llm_simulation/data/processed_data/community_{COMM}/"
raw_data_folder = f"/home/fanqi/llm_simulation/data/raw_data/community_{COMM}/"
generator_device = 'cuda:1'
other_device = "cuda:0"

with open("/home/fanqi/llm_simulation/config.json", "r") as config_file:
    config = json.load(config_file)

tweets_df = pd.read_json(raw_data_folder + "tweet.json")
with open(processed_data_folder + "id_tweet.json", "r", encoding="utf-8") as f:
    all_tweets = json.load(f)

users_df = pd.read_json(raw_data_folder + "user_summary.json")
labels_df = pd.read_csv(raw_data_folder + "label.csv")
user_ids = labels_df['id'].values
uid_index={uid:index for index,uid in enumerate(user_ids)}
bot_ids = labels_df[labels_df['label'] == 'bot']['id'].tolist()
bot_indices = [uid_index[key] for key in bot_ids if key in uid_index]
human_ids = labels_df[labels_df['label'] == 'human']['id'].tolist()
indexed_human_ids = list(enumerate(human_ids))
batch_size = config["generator_training"]["batch_size"]
training_epoch = config["generator_training"]["epoch"]

dataset=Twibot22(root=processed_data_folder,device=other_device,process=False,save=False)

des_tensor,tweets_tensor,num_prop,category_prop,edge_index,edge_type,labels,train_idx,val_idx,test_idx=dataset.dataloader(tweets_path="tweets_tensor_dpo_0.pt")

inputs=[des_tensor, tweets_tensor, num_prop, category_prop, edge_index, edge_type, labels, train_idx, val_idx, test_idx]
# model_path = 'NousResearch/Llama-2-7b-chat-hf'
model_discriminator = BotRGCN(cat_prop_size=config["discriminator"]["cat_prop_size"],
                            embedding_dimension=config['discriminator']["embedding_dimension"]).to(other_device)
optimizer_discriminator = torch.optim.AdamW(model_discriminator.parameters(),
                        lr=config["discriminator"]["lr"],weight_decay=config["discriminator"]["weight_decay"])
loss_func_discriminator = nn.CrossEntropyLoss()

'''raw data pretraining'''

if __name__ == '__main__':
    all_inputs = {generator_version: inputs.copy()}

    '''Pretrain discriminator on raw data'''

    print("\n=== raw data detector ===\n")
    model_discriminator = BotRGCN(cat_prop_size=config["discriminator"]["cat_prop_size"],
                            embedding_dimension=config['discriminator']["embedding_dimension"]).to(other_device)
    optimizer_discriminator = torch.optim.AdamW(model_discriminator.parameters(),
                            lr=config["discriminator"]["lr"],weight_decay=config["discriminator"]["weight_decay"])
    loss_func_discriminator = nn.CrossEntropyLoss()
    train_discrim(model=model_discriminator,
                loss_func=loss_func_discriminator,
                optimizer=optimizer_discriminator,
                epochs=config["discriminator"]["pretrain_epochs"],
                inputs=all_inputs)
    # torch.save(model_discriminator.state_dict(), detector_folder + f'origin_bot_dataset_comm_{COMM}.pth')
    test_discrim(model=model_discriminator,
                loss_func=loss_func_discriminator,
                inputs=all_inputs)
    
    '''======================'''
    '''Start main training loop'''

    raw_tweet_tensor = torch.load(processed_data_folder + "tweets_tensor.pt", weights_only=True).to(other_device)


    print("\n=== Start main training loop! ===\n")
    for epoch in range(training_epoch):
        print(f"=== epoch: {epoch} ===")
        
        '''update tweets tensor'''
        updated_tweets_tensor_path = processed_data_folder + f"tweets_tensor_dpo_{epoch}.pt"
        updated_tweets_tensor=torch.load(updated_tweets_tensor_path, weights_only=True).to(other_device)
        inputs[1] = updated_tweets_tensor
        all_inputs[f"DPO{epoch} bot"] = inputs.copy()

        '''train detector'''
        print(f"\n=== detector training: {epoch} ===\n")
        # model_discriminator = BotRGCN(cat_prop_size=config["discriminator"]["cat_prop_size"],
        #                     embedding_dimension=config['discriminator']["embedding_dimension"]).to(other_device)
        # optimizer_discriminator = torch.optim.AdamW(model_discriminator.parameters(),
        #                 lr=config["discriminator"]["lr"],weight_decay=config["discriminator"]["weight_decay"])
        train_discrim(model=model_discriminator,
                loss_func=loss_func_discriminator,
                optimizer=optimizer_discriminator,
                epochs=config["discriminator"]["pretrain_epochs"],
                inputs=all_inputs)
        test_discrim(model=model_discriminator,
                    loss_func=loss_func_discriminator,
                    inputs=all_inputs)
        # exit()
        
        '''sample dpo dataset'''
        dpo_data_path = f"{dpo_dataset_path}/community_{COMM}/dpo_dataset_{epoch}.jsonl"
        if not os.path.exists(dpo_data_path):
            samples_per_gpu = int(dpo_dataset_size/batch_size) // num_gpus
            manager = Manager()
            queue = manager.Queue()
            processes = []

            for i, gpu_id in enumerate(free_gpus):
                p = Process(target=generate_data, args=(gpu_id, queue, samples_per_gpu, model_path, model_discriminator, inputs))
                processes.append(p)
                p.start()
            
            for p in processes:
                p.join()

            dpo_dataset = []
            while not queue.empty():
                dpo_dataset.extend(queue.get())
            with open(dpo_data_path, "w") as f:
                json.dump(dpo_dataset, f)
            print(f"==== Successfully build dpo dataset! ====")

        ''' DPO training '''
        dpo_model_path = f"/home/fanqi/llm_simulation/models/DPO/community_{COMM}/merged_ckp_{epoch}"
        if not os.path.exists(dpo_model_path):
            model_args = {
                "torch_dtype": PRECISION_MAPPING[config["llm_model"]["precision_bits"]],
                "device_map": {"": generator_device}
            }
            model_generator = Generator(model_path, **model_args)
            model_generator.model = get_peft_model(model_generator.model, peft_config)
            dpo_train_dataset = load_dataset("json", data_files=dpo_data_path, split="train")
            DPO_trainer = DPOTrainer(model=model_generator.model, 
                                    args=DPO_config, 
                                    tokenizer=model_generator.tokenizer, 
                                    train_dataset=dpo_train_dataset,
                                    beta=0.15,
                                    max_length=2048
                                    )
            DPO_trainer.train()
            DPO_trainer.save_model(f"/home/fanqi/llm_simulation/models/DPO/community_{COMM}/result_{epoch}")
            # DPO_trainer.model.save_pretrained(f"/home/fanqi/llm_simulation/models/DPO/DPO_ckp_{epoch}")
            # model_generator.tokenizer.save_pretrained(f"/home/fanqi/llm_simulation/models/DPO/DPO_ckp_{epoch}")

            # update generator
            merge_peft_adapters(adapter_dir=f"/home/fanqi/llm_simulation/models/DPO/community_{COMM}/result_{epoch}", output_path=f"/home/fanqi/llm_simulation/models/DPO/community_{COMM}/merged_ckp_{epoch}")
        model_path = dpo_model_path
        # exit()

