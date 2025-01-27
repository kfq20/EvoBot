import torch
from torch import nn
import json
from models.discriminator import BotRGCN, train_discrim, test_discrim
from Dataset import Twibot22
import random
from tqdm import tqdm
import os
import numpy as np
os.environ["WANDB_DISABLED"]="true"
SEED = 42
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 设置随机种子
set_seed(SEED)
COMM = 5
detector_version = "raw bot"

processed_data_folder=f"/home/fanqi/llm_simulation/data/processed_data/community_{COMM}/"
detector_folder = f"/home/fanqi/llm_simulation/models/Detector/community_{COMM}/"

other_device = "cuda:4"
with open("/home/fanqi/llm_simulation/config.json", "r") as config_file:
    config = json.load(config_file)

bot_dataset = Twibot22(root=processed_data_folder, device=other_device,process=False,save=False)
tweets_path = "tweets_tensor.pt" if "dpo" not in detector_version else f"tweets_tensor_{detector_version}.pt"
des_tensor1,tweets_tensor1,num_prop1,category_prop1,edge_index1,edge_type1,labels1,train_idx1,val_idx1,test_idx1=bot_dataset.dataloader(tweets_path=tweets_path)

# inputs=[des_tensor, tweets_tensor, num_prop, category_prop, edge_index, edge_type, labels, train_idx, val_idx, test_idx]
bot_inputs = [des_tensor1, tweets_tensor1, num_prop1, category_prop1, edge_index1, edge_type1, labels1, train_idx1, val_idx1, test_idx1]

if __name__ == '__main__':
    all_inputs = {}
    all_inputs[f"{detector_version}"] = bot_inputs.copy()
    print(f"\n=== {detector_version} bot ===\n")
    
    model_discriminator = BotRGCN(cat_prop_size=config["discriminator"]["cat_prop_size"],
                            embedding_dimension=config['discriminator']["embedding_dimension"]).to(other_device)
    optimizer_discriminator = torch.optim.AdamW(model_discriminator.parameters(),
                            lr=config["discriminator"]["lr"],weight_decay=config["discriminator"]["weight_decay"])
    loss_func_discriminator = nn.CrossEntropyLoss()
    if "dpo" in detector_version:
        raw_tweet_tensor = torch.load(processed_data_folder + "tweets_tensor.pt", weights_only=True).to(other_device)
        bot_inputs[1] = raw_tweet_tensor
        all_inputs["raw data"] = bot_inputs.copy()
        for i in range(int(detector_version[-1])):
            if i == int(detector_version[-1]):
                continue
            updated_tweets_tensor_path = processed_data_folder + f"tweets_tensor_dpo_{i}.pt"
            updated_tweets_tensor=torch.load(processed_data_folder + f"tweets_tensor_dpo_{i}.pt", weights_only=True).to(other_device)
            bot_inputs[1] = updated_tweets_tensor
            all_inputs[f"dpo {i}"] = bot_inputs.copy()
    train_discrim(model=model_discriminator,
                loss_func=loss_func_discriminator,
                optimizer=optimizer_discriminator,
                epochs=config["discriminator"]["pretrain_epochs"],
                inputs=all_inputs)
    # torch.save(model_discriminator.state_dict(), detector_folder + f'{detector_version}.pth')
    test_discrim(model=model_discriminator,
                loss_func=loss_func_discriminator,
                inputs={f"{detector_version}": all_inputs[detector_version]})
    
    vanilla_tweets_tensor_path = processed_data_folder + f"tweets_tensor_vanilla.pt"
    vanilla_tweets_tensor=torch.load(vanilla_tweets_tensor_path, weights_only=True).to(other_device)
    bot_inputs[1] = vanilla_tweets_tensor
    test_discrim(model=model_discriminator,
                    loss_func=loss_func_discriminator,
                    inputs={f"vanilla llama": bot_inputs})


    if "dpo" in detector_version:
        print(f"\n=== raw bot ===\n")
        raw_tweet_tensor = torch.load(processed_data_folder + "tweets_tensor.pt", weights_only=True).to(other_device)
        bot_inputs[1] = raw_tweet_tensor
        test_discrim(model=model_discriminator,
                    loss_func=loss_func_discriminator,
                    inputs={f"raw bot": bot_inputs})
        
    for epoch in [0,1,2,3,4]:
        if "dpo" in detector_version and epoch == int(detector_version[-1]):
            continue
        print(f"=== epoch: {epoch} ===")
        
        '''update tweets tensor'''
        updated_tweets_tensor_path = processed_data_folder + f"tweets_tensor_dpo_{epoch}.pt"
        updated_tweets_tensor=torch.load(processed_data_folder + f"tweets_tensor_dpo_{epoch}.pt", weights_only=True).to(other_device)
        bot_inputs[1] = updated_tweets_tensor

        print(f"\n=== dpo {epoch} bot ===\n")
        test_discrim(model=model_discriminator,
                    loss_func=loss_func_discriminator,
                    inputs={f"DPO {epoch}": bot_inputs})
