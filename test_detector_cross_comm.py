from transformers import pipeline
import torch
from torch import nn
import json
from models.discriminator import BotRGCN, train_discrim, test_discrim
from Dataset import Twibot22

PRECISION_MAPPING = {16: torch.float16, 32: torch.float32, 64: torch.float64}
COMM = 5
test_comm_num = 5
processed_data_folder=f"/home/fanqi/llm_simulation/data/processed_data/community_{COMM}/"
device = "cuda:3"

with open("/home/fanqi/llm_simulation/config.json", "r") as config_file:
    config = json.load(config_file)

datasets = []
all_inputs = []
for i in range(test_comm_num):
    i_processed_data_folder = f"/home/fanqi/llm_simulation/data/processed_data/community_{i}/"
    bot_dataset = Twibot22(root=i_processed_data_folder, device=device,process=False,save=False)
    datasets.append(bot_dataset)
    des_tensor,tweets_tensor,num_prop,category_prop,edge_index,edge_type,labels,train_idx,val_idx,test_idx=bot_dataset.dataloader(tweets_path="tweets_tensor.pt")
    input_tensor = [des_tensor,tweets_tensor,num_prop,category_prop,edge_index,edge_type,labels,train_idx,val_idx,test_idx]
    all_inputs.append(input_tensor)

loss_func_discriminator = nn.CrossEntropyLoss()

if __name__ == '__main__':
    # torch.save(model_discriminator.state_dict(), detector_folder + f'origin_bot_dataset_comm_{COMM}.pth')
    print("\n=== Test Original Detector ===\n")

    model_discriminator = BotRGCN(cat_prop_size=config["discriminator"]["cat_prop_size"],
                            embedding_dimension=config['discriminator']["embedding_dimension"]).to(device)
    optimizer_discriminator = torch.optim.AdamW(model_discriminator.parameters(),
                            lr=config["discriminator"]["lr"],weight_decay=config["discriminator"]["weight_decay"])
    train_discrim(model=model_discriminator,
                loss_func=loss_func_discriminator,
                optimizer=optimizer_discriminator,
                epochs=config["discriminator"]["pretrain_epochs"],
                inputs=all_inputs[COMM])
    for test_comm in range(test_comm_num):
        test_discrim(model=model_discriminator,
                loss_func=loss_func_discriminator,
                inputs=all_inputs[test_comm])
    
    print("\n=== Test Final Detector ===\n")

    updated_tweets_tensor=torch.load(processed_data_folder + "tweets_dpo_4_tensor.pt", weights_only=True).to(device)
    all_inputs[COMM][1] = updated_tweets_tensor
    train_discrim(model=model_discriminator,
                loss_func=loss_func_discriminator,
                optimizer=optimizer_discriminator,
                epochs=config["discriminator"]["pretrain_epochs"],
                inputs=all_inputs[COMM])
    
    for test_comm in range(test_comm_num):
        test_discrim(model=model_discriminator,
                loss_func=loss_func_discriminator,
                inputs=all_inputs[test_comm])
