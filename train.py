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
from utils.Dataset import Twibot22
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

os.environ["WANDB_DISABLED"]="true"
# os.environ["WANDB_DISABLED"]="true"

PRECISION_MAPPING = {16: torch.float16, 32: torch.float32, 64: torch.float64}
COMM = 11
processed_data_folder=f"./data/processed_data/community_{COMM}/"
bot_processed_data = f"./data/processed_data/community_{COMM}/"
raw_data_folder = f"./data/raw_data/community_{COMM}/"
detector_folder = "./models/Detector/"
generator_device = 'cuda:1'
other_device = "cuda:0"

with open("./config.json", "r") as config_file:
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
DPO_config = DPOConfig(**config["DPO_trainer"], bf16=True, gradient_checkpointing=True)
peft_config = LoraConfig(**config["peft"])
feature_extract=pipeline('feature-extraction',model='roberta-base', device=other_device, tokenizer='roberta-base', truncation=True,padding='max_length', add_special_tokens = True, max_length=512)
feature_extract.tokenizer.model_max_length = 512
dpo_dataset_size = config["DPO_dataset_size"]
batch_size = config["generator_training"]["batch_size"]
training_epoch = config["generator_training"]["epoch"]

dataset=Twibot22(root=processed_data_folder,device=other_device,process=False,save=False)
bot_dataset = Twibot22(root=bot_processed_data, device=other_device,process=False,save=False)

des_tensor,tweets_tensor,num_prop,category_prop,edge_index,edge_type,labels,train_idx,val_idx,test_idx=dataset.dataloader(tweets_path="tweets_tensor.pt")

inputs=[des_tensor, tweets_tensor, num_prop, category_prop, edge_index, edge_type, labels, train_idx, val_idx, test_idx]
model_path = f"./models/SFT/sft_merged_ckp_{COMM}"
# model_path = 'NousResearch/Llama-2-7b-chat-hf'
dpo_dataset_path = "./data/dpo_data"
model_discriminator = BotRGCN(cat_prop_size=config["discriminator"]["cat_prop_size"],
                            embedding_dimension=config['discriminator']["embedding_dimension"]).to(other_device)
optimizer_discriminator = torch.optim.AdamW(model_discriminator.parameters(),
                        lr=config["discriminator"]["lr"],weight_decay=config["discriminator"]["weight_decay"])
loss_func_discriminator = nn.CrossEntropyLoss()

def build_DPO_data(prompts, response_1, response_2, score_1, score_2):
    '''score: [batch, 2], where the first dim is the logits of human label'''
    normalized_score_1 = F.softmax(score_1, dim=1)
    normalized_score_2 = F.softmax(score_2, dim=1)
    filtered_data = []
    for i, prompt in enumerate(prompts):
        filtered_data.append({
            "prompt": prompt,
            "chosen": response_1[i] if normalized_score_1[i][0] >= normalized_score_2[i][0] else response_2[i],
            "rejected": response_2[i] if normalized_score_2[i][0] <= normalized_score_1[i][0] else response_1[i]
        })
    return filtered_data

def test_generate_data(batch_prompt, model_path, device):
    model_args = {
        # "torch_dtype": PRECISION_MAPPING[config["llm_model"]["precision_bits"]],
        "torch_dtype": torch.float16,
        "device_map": {"": device}
    }
    model_generator = Generator(model_path, **model_args)
    model_generator.model = get_peft_model(model_generator.model, peft_config)
    batch_response_1 = model_generator.generate_text(batch_prompt, max_length=1024, temperature=0.7, do_sample=True, repetition_penalty=1.3)
    processed_outputs = []
    for output in batch_response_1:
        response_output = re.sub(r'\[INST\].*?\[/INST\]', '', output, flags=re.DOTALL).strip()
        processed_outputs.append(response_output)
    print(f"=== prompt ===\n{batch_prompt}\n")
    print(f"=== response ===\n{processed_outputs}")

def generate_data(gpu_id, queue, samples, model_path, model_discriminator, inputs):
    torch.cuda.set_device(gpu_id)
    device = f"cuda:{gpu_id}"
    model_args = {
        "torch_dtype": PRECISION_MAPPING[config["llm_model"]["precision_bits"]],
        "device_map": {"": device}
    }
    model_generator = Generator(model_path, **model_args)
    model_generator.model = get_peft_model(model_generator.model, peft_config)
    url_pattern1 = re.compile(r'https?://\S+')
    url_pattern2 = re.compile(r'http?://\S+')
    dpo_dataset = []
    for _ in tqdm(range(samples)):
        sample_bot_ids = random.sample(bot_ids, batch_size)
        sample_bot_indices = [uid_index[key] for key in sample_bot_ids]
        batch_prompt = []
        for bot_id in sample_bot_ids:
            neighbor_ids = find_user_neighbors(raw_data_folder + "edge.csv", bot_id, 5)
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
            batch_response_1 = model_generator.generate_text(batch_prompt, max_length=2048, temperature=0.7, do_sample=True, repetition_penalty=1.3)
            batch_response_2 = model_generator.generate_text(batch_prompt, max_length=2048, temperature=0.7, do_sample=True, repetition_penalty=1.3)
            response_tensor = []
            for each_response in batch_response_1 + batch_response_2:
                if re.search(r'\n', each_response):
                    segments = re.split(r'\n', each_response.strip())
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
                    # segments = [text.strip()]
                    results = [each_response]
                if len(results) == 0:
                    total_each_person_tweets=torch.zeros(768)
                else:
                    all_tweet_tensor=torch.tensor(feature_extract(results, padding=True, max_length=512, truncation=True, batch_size=len(results)))
                    total_word_tensor = torch.mean(all_tweet_tensor, dim=2)
                    total_each_person_tweets = torch.mean(total_word_tensor, dim=0)
                response_tensor.append(total_each_person_tweets)

            response_tensor = torch.cat(response_tensor, dim=0).to(other_device)
            # response_tensor = feature_extractor.feature_extract(batch_response_1+batch_response_2).to(other_device)
            tweets_tensor = inputs[1]
            tweets_tensor_1 = tweets_tensor.detach()
            tweets_tensor_1[sample_bot_indices] = response_tensor[:batch_size]
            tweets_tensor_2 = tweets_tensor.detach()
            tweets_tensor_2[sample_bot_indices] = response_tensor[batch_size:]
            output_1 = model_discriminator(des_tensor,
                                    tweets_tensor_1,
                                    num_prop,
                                    category_prop,
                                    edge_index,
                                    edge_type)
            output_2 = model_discriminator(des_tensor,
                                    tweets_tensor_2,
                                    num_prop,
                                    category_prop,
                                    edge_index,
                                    edge_type)
            response_score_1 = output_1[sample_bot_indices]
            response_score_2 = output_2[sample_bot_indices]
        
        dpo_data = build_DPO_data(batch_prompt, batch_response_1, batch_response_2, response_score_1, response_score_2)
        dpo_dataset.extend(dpo_data)
    queue.put(dpo_dataset)

if __name__ == '__main__':
    all_inputs = {"original bot": inputs.copy()}

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
    '''train on vanilla llama'''

    replace_indexed_human_ids = random.sample(indexed_human_ids, 500)

    replaced_human_indices, replaced_human_ids = zip(*replace_indexed_human_ids)
    print("Vanilla LLM results")
    set_start_method('spawn', force=True) 
    free_gpus = get_free_gpus()
    print(f"Free GPUs found: {free_gpus}")
    num_gpus = len(free_gpus)
    vanilla_tweet_path = processed_data_folder + "id_tweet_vanilla.json"
    if not os.path.exists(vanilla_tweet_path):
        bot_ids_per_gpu = len(bot_ids) // num_gpus 
        manager = Manager()
        queue = manager.Queue()
        processes = []
        for i, gpu_id in enumerate(free_gpus):
            start_idx = i * bot_ids_per_gpu
            end_idx = start_idx + bot_ids_per_gpu
            if i != num_gpus - 1:
                bot_ids_gpu = bot_ids[start_idx:end_idx]
            else:
                bot_ids_gpu = bot_ids[start_idx:]
            p = Process(target=rewrite_all_bot_tweets, args=(gpu_id, "NousResearch/Llama-2-7b-chat-hf", bot_ids_gpu, COMM, queue))
            processes.append(p)
            p.start()
        for p in processes:
            p.join()
        all_new_tweets = [None] * num_gpus
        while not queue.empty():
            gpu_id, responses = queue.get()
            all_new_tweets[free_gpus.index(gpu_id)] = responses
        all_new_tweets = [response for responses in all_new_tweets for response in responses]
        print(f"\n===== REPLACED TWEET NUM: {len(all_new_tweets)}======\n")
        
        replaced_tweet_dict = {tweet["author_id"]: tweet["text"] for tweet in all_new_tweets}
        replaced_tweets = []
        for user_index in all_tweets:
            if int(user_index) in bot_indices: # replaced bot
                replaced_tweets.append({user_index: replaced_tweet_dict[user_ids[int(user_index)]]})
            else:
                replaced_tweets.append({user_index: all_tweets[user_index]})
        id_replaced_tweets = {k: v for d in replaced_tweets for k, v in d.items()}
        with open(vanilla_tweet_path, "w", encoding="utf-8") as f:
            json.dump(id_replaced_tweets, f, indent=4, ensure_ascii=False)

    # process replaced tweet
    vanilla_tweets_tensor_path = processed_data_folder + "tweets_tensor_vanilla.pt"
    if not os.path.exists(vanilla_tweets_tensor_path):
        tweets_embedding(each_user_tweets_path=vanilla_tweet_path,
                          output_path=vanilla_tweets_tensor_path,
                          community=COMM, device=other_device)
    updated_tweets_tensor=torch.load(vanilla_tweets_tensor_path).to(other_device)
    inputs[1] = updated_tweets_tensor
    
    '''======================'''
    '''Start main training loop'''

    print("\n=== Start main training loop! ===\n")
    print(f"\n=== Current model {model_path} ===\n")
    set_start_method('spawn', force=True) 
    free_gpus = get_free_gpus()
    print(f"Free GPUs found: {free_gpus}")
    num_gpus = len(free_gpus)
    for epoch in range(training_epoch):
        print(f"=== epoch: {epoch} ===")

        '''use the sft model or dpo model to rewrite the replaced human tweet'''
        dpo_replaced_tweet_path = processed_data_folder + f"id_tweet_dpo_{epoch}.json"
        if not os.path.exists(dpo_replaced_tweet_path):
            bot_ids_per_gpu = len(bot_ids) // num_gpus 
            manager = Manager()
            queue = manager.Queue()
            processes = []
            for i, gpu_id in enumerate(free_gpus):
                start_idx = i * bot_ids_per_gpu
                end_idx = start_idx + bot_ids_per_gpu
                if i != num_gpus - 1:
                    bot_ids_gpu = bot_ids[start_idx:end_idx]
                else:
                    bot_ids_gpu = bot_ids[start_idx:]
                p = Process(target=rewrite_all_bot_tweets, args=(gpu_id, model_path, bot_ids_gpu, COMM, queue))
                processes.append(p)
                p.start()
            for p in processes:
                p.join()
            all_new_tweets = [None] * num_gpus
            while not queue.empty():
                gpu_id, responses = queue.get()
                all_new_tweets[free_gpus.index(gpu_id)] = responses
            all_new_tweets = [response for responses in all_new_tweets for response in responses]

            replaced_tweet_dict = {tweet["author_id"]: tweet["text"] for tweet in all_new_tweets}
            replaced_tweets = []
            for user_index in all_tweets:
                if int(user_index) in bot_indices: # replaced human
                    replaced_tweets.append({user_index: replaced_tweet_dict[user_ids[int(user_index)]]})
                else:
                    replaced_tweets.append({user_index: all_tweets[user_index]})

            id_replaced_tweets = {k: v for d in replaced_tweets for k, v in d.items()}
            with open(dpo_replaced_tweet_path, "w", encoding="utf-8") as f:
                json.dump(id_replaced_tweets, f, indent=4, ensure_ascii=False)
        
        '''update tweets tensor'''
        updated_tweets_tensor_path = processed_data_folder + f"tweets_tensor_dpo_{epoch}.pt"
        if not os.path.exists(updated_tweets_tensor_path):
            tweets_embedding(each_user_tweets_path=dpo_replaced_tweet_path, 
                             output_path=updated_tweets_tensor_path,
                             community=COMM, dpo_epoch=epoch, device=other_device)
        updated_tweets_tensor=torch.load(updated_tweets_tensor_path, weights_only=True).to(other_device)
        inputs[1] = updated_tweets_tensor
        all_inputs[f"DPO{epoch} bot"] = inputs.copy()

        '''train detector'''
        print(f"\n=== detector training: {epoch} ===\n")
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
        if not os.path.exists(f"{dpo_dataset_path}/community_{COMM}/"):
            os.makedirs(f"{dpo_dataset_path}/community_{COMM}/")
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
        dpo_model_path = f"./models/DPO/community_{COMM}/merged_ckp_{epoch}"
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
            DPO_trainer.save_model(f"./models/DPO/community_{COMM}/result_{epoch}")
            DPO_trainer.model.save_pretrained(f"./models/DPO/DPO_ckp_{epoch}")
            model_generator.tokenizer.save_pretrained(f"./models/DPO/DPO_ckp_{epoch}")

            # update generator
            merge_peft_adapters(adapter_dir=f"./models/DPO/community_{COMM}/result_{epoch}", output_path=dpo_model_path)
        model_path = dpo_model_path
        # exit()

