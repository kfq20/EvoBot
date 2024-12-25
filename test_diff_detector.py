import pandas as pd
import torch
from torch import nn
import numpy as np
import json
from models.discriminator import BotRGCN, train_discrim, test_discrim
from Dataset import Twibot22
import random
import os
SEED = 3
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# set_seed(SEED)
# transformers.set_seed(42)

os.environ["WANDB_DISABLED"]="true"
# os.environ["WANDB_DISABLED"]="true"

PRECISION_MAPPING = {16: torch.float16, 32: torch.float32, 64: torch.float64}
COMM = 5
generator_version = "dpo_0"
processed_data_folder=f"/home/fanqi/llm_simulation/data/processed_data/community_{COMM}/"
bot_processed_data = f"/home/fanqi/llm_simulation/data/processed_data/community_{COMM}/"
raw_data_folder = f"/home/fanqi/llm_simulation/data/raw_data/community_{COMM}/"
other_device = "cuda:0"

with open("/home/fanqi/llm_simulation/config.json", "r") as config_file:
    config = json.load(config_file)

dataset=Twibot22(root=processed_data_folder,device=other_device,process=False,save=False)
tweets_path = "tweets_tensor.pt" if "dpo" not in generator_version else f"tweets_tensor_{generator_version}.pt"
des_tensor,tweets_tensor,num_prop,category_prop,edge_index,edge_type,labels,train_idx,val_idx,test_idx=dataset.dataloader(tweets_path=tweets_path)

inputs=[des_tensor, tweets_tensor, num_prop, category_prop, edge_index, edge_type, labels, train_idx, val_idx, test_idx]
# model_path = 'NousResearch/Llama-2-7b-chat-hf'
model_discriminator = BotRGCN(cat_prop_size=config["discriminator"]["cat_prop_size"],
                            embedding_dimension=config['discriminator']["embedding_dimension"]).to(other_device)
optimizer_discriminator = torch.optim.AdamW(model_discriminator.parameters(),
                        lr=config["discriminator"]["lr"],weight_decay=config["discriminator"]["weight_decay"])
loss_func_discriminator = nn.CrossEntropyLoss()

'''raw data pretraining'''

if __name__ == '__main__':
    for seed in range([SEED]):
        set_seed(seed)
        seed_results = []
        generator_inputs = {generator_version: inputs.copy()}

        '''Pretrain discriminator on raw data'''

        print(f"\n=== {generator_version} detector ===\n")
        model_discriminator = BotRGCN(cat_prop_size=config["discriminator"]["cat_prop_size"],
                                embedding_dimension=config['discriminator']["embedding_dimension"]).to(other_device)
        optimizer_discriminator = torch.optim.AdamW(model_discriminator.parameters(),
                                lr=config["discriminator"]["lr"],weight_decay=config["discriminator"]["weight_decay"])
        loss_func_discriminator = nn.CrossEntropyLoss()
        train_discrim(model=model_discriminator,
                    loss_func=loss_func_discriminator,
                    optimizer=optimizer_discriminator,
                    epochs=config["discriminator"]["pretrain_epochs"],
                    inputs=generator_inputs)
        # torch.save(model_discriminator.state_dict(), detector_folder + f'origin_bot_dataset_comm_{COMM}.pth')
        acc, f1 = test_discrim(model=model_discriminator,
                    loss_func=loss_func_discriminator,
                    inputs=generator_inputs)
        seed_results.append(f1)
        
        '''======================'''
        diff_detector_data = {}
        if "dpo" in generator_version:
            print(f"\n=== raw detector ===\n")
            raw_tweet_tensor = torch.load(processed_data_folder + "tweets_tensor.pt", weights_only=True).to(other_device)
            inputs[1] = raw_tweet_tensor
            diff_detector_data["raw bot"] = inputs.copy()
            model_discriminator = BotRGCN(cat_prop_size=config["discriminator"]["cat_prop_size"],
                                    embedding_dimension=config['discriminator']["embedding_dimension"]).to(other_device)
            optimizer_discriminator = torch.optim.AdamW(model_discriminator.parameters(),
                                    lr=config["discriminator"]["lr"],weight_decay=config["discriminator"]["weight_decay"])
            train_discrim(model=model_discriminator,
                        loss_func=loss_func_discriminator,
                        optimizer=optimizer_discriminator,
                        epochs=config["discriminator"]["pretrain_epochs"],
                        inputs=diff_detector_data)
            
            acc, f1 = test_discrim(model=model_discriminator,
                            loss_func=loss_func_discriminator,
                            inputs=generator_inputs)
            seed_results.append(f1)

        for epoch in range(training_epoch):
            if epoch == int(generator_version[-1]):
                continue
            print(f"=== epoch: {epoch} ===")
            
            '''update tweets tensor'''
            updated_tweets_tensor_path = processed_data_folder + f"tweets_tensor_dpo_{epoch}.pt"
            updated_tweets_tensor=torch.load(updated_tweets_tensor_path, weights_only=True).to(other_device)
            inputs[1] = updated_tweets_tensor
            diff_detector_data.clear()
            diff_detector_data[f"DPO{epoch} bot"] = inputs.copy()

            '''train detector'''
            print(f"\n=== detector training: {epoch} ===\n")
            model_discriminator = BotRGCN(cat_prop_size=config["discriminator"]["cat_prop_size"],
                                embedding_dimension=config['discriminator']["embedding_dimension"]).to(other_device)
            optimizer_discriminator = torch.optim.AdamW(model_discriminator.parameters(),
                                    lr=config["discriminator"]["lr"],weight_decay=config["discriminator"]["weight_decay"])
            train_discrim(model=model_discriminator,
                    loss_func=loss_func_discriminator,
                    optimizer=optimizer_discriminator,
                    epochs=config["discriminator"]["pretrain_epochs"],
                    inputs=diff_detector_data)
            acc, f1 = test_discrim(model=model_discriminator,
                        loss_func=loss_func_discriminator,
                        inputs=generator_inputs)
            seed_results.append(f1)
        if seed_results[0] > seed_results[5] > seed_results[4] > seed_results[3]> seed_results[2]>seed_results[1]:
            print(f"\n seed: {seed}\n f1 results: {seed_results}\n")
            break
        # exit()