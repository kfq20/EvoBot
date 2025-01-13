import torch
from torch import nn

import json
from models.discriminator import BotRGCN, train_discrim, test_discrim
from Dataset import Twibot22
from data.raw_data.preprocess import extract_tweets_1, tweets_embedding
import os

os.environ["WANDB_DISABLED"]="true"
# os.environ["WANDB_DISABLED"]="true"

PRECISION_MAPPING = {16: torch.float16, 32: torch.float32, 64: torch.float64}
COMM = 5
processed_data_folder=f"/home/fanqi/llm_simulation/data/processed_data/community_{COMM}/"
bot_processed_data = f"/home/fanqi/llm_simulation/data/processed_data/community_{COMM}/"
raw_data_folder = f"/home/fanqi/llm_simulation/data/raw_data/community_{COMM}/"
other_device = "cuda:0"

with open("/home/fanqi/llm_simulation/config.json", "r") as config_file:
    config = json.load(config_file)

dataset=Twibot22(root=processed_data_folder,device=other_device,process=False,save=False)

des_tensor,tweets_tensor,num_prop,category_prop,edge_index,edge_type,labels,train_idx,val_idx,test_idx=dataset.dataloader(tweets_path="tweets_tensor_dpo_4.pt")

inputs=[des_tensor, tweets_tensor, num_prop, category_prop, edge_index, edge_type, labels, train_idx, val_idx, test_idx]

if __name__ == '__main__':
    all_inputs = {"final": inputs.copy()}

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
    exit()
    gpt_tweet_path = processed_data_folder + "id_tweet_dpo_4_wo_sft.json"
    gpt_tweets_tensor_path = processed_data_folder + "tweets_tensor_dpo_4_wo_sft.pt"
    if not os.path.exists(gpt_tweets_tensor_path):
        tweets_embedding(each_user_tweets_path=gpt_tweet_path,
                          output_path=gpt_tweets_tensor_path,
                          community=COMM, device=other_device)
    gpt_tweets_tensor=torch.load(gpt_tweets_tensor_path).to(other_device)
    inputs[1] = gpt_tweets_tensor
    test_discrim(model=model_discriminator,
                loss_func=loss_func_discriminator,
                inputs={"gpt_bot":inputs})