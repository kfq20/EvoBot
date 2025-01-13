from transformers import pipeline
import torch
from torch import nn
import json
from models.discriminator import BotRGCN, train_discrim, test_discrim
from Dataset import Twibot22
import random
import numpy as np

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

set_seed(SEED)

PRECISION_MAPPING = {16: torch.float16, 32: torch.float32, 64: torch.float64}
COMM = 8
detector_version = "original"
test_comm_num = 12
processed_data_folder=f"/home/fanqi/llm_simulation/data/processed_data/community_{COMM}/"
device = "cuda:7"

with open("/home/fanqi/llm_simulation/config.json", "r") as config_file:
    config = json.load(config_file)

bot_dataset = Twibot22(root=processed_data_folder, device=device,process=False,save=False)
tweets_path = "tweets_tensor.pt" if "dpo" not in detector_version else f"tweets_tensor_{detector_version}.pt"
des_tensor1,tweets_tensor1,num_prop1,category_prop1,edge_index1,edge_type1,labels1,train_idx1,val_idx1,test_idx1=bot_dataset.dataloader(tweets_path=tweets_path)

# inputs=[des_tensor, tweets_tensor, num_prop, category_prop, edge_index, edge_type, labels, train_idx, val_idx, test_idx]
bot_inputs = [des_tensor1, tweets_tensor1, num_prop1, category_prop1, edge_index1, edge_type1, labels1, train_idx1, val_idx1, test_idx1]

test_inputs = {}
for i in range(test_comm_num):
    i_processed_data_folder = f"/home/fanqi/llm_simulation/data/processed_data/community_{i}/"
    dataset = Twibot22(root=i_processed_data_folder, device=device,process=False,save=False)
    des_tensor,tweets_tensor,num_prop,category_prop,edge_index,edge_type,labels,train_idx,val_idx,test_idx=dataset.dataloader(tweets_path="tweets_tensor.pt")
    input_tensor = [des_tensor,tweets_tensor,num_prop,category_prop,edge_index,edge_type,labels,train_idx,val_idx,test_idx]
    test_inputs[f"comm {i}"] = input_tensor

loss_func_discriminator = nn.CrossEntropyLoss()

if __name__ == '__main__':
    all_inputs = {}
    all_inputs[f"{detector_version}"] = bot_inputs.copy()
    # torch.save(model_discriminator.state_dict(), detector_folder + f'origin_bot_dataset_comm_{COMM}.pth')
    if "dpo" in detector_version:
        raw_tweet_tensor = torch.load(processed_data_folder + "tweets_tensor.pt", weights_only=True).to(device)
        bot_inputs[1] = raw_tweet_tensor
        all_inputs["raw data"] = bot_inputs.copy()
        for i in range(int(detector_version[-1])):
            if i == int(detector_version[-1]):
                continue
            updated_tweets_tensor_path = processed_data_folder + f"tweets_tensor_dpo_{i}.pt"
            updated_tweets_tensor=torch.load(processed_data_folder + f"tweets_tensor_dpo_{i}.pt", weights_only=True).to(device)
            bot_inputs[1] = updated_tweets_tensor
            all_inputs[f"dpo {i}"] = bot_inputs.copy()

    print(f"\n=== Detector: {detector_version}; Trained Community: {COMM} ===\n")

    model_discriminator = BotRGCN(cat_prop_size=config["discriminator"]["cat_prop_size"],
                            embedding_dimension=config['discriminator']["embedding_dimension"]).to(device)
    optimizer_discriminator = torch.optim.AdamW(model_discriminator.parameters(),
                            lr=config["discriminator"]["lr"],weight_decay=config["discriminator"]["weight_decay"])
    train_discrim(model=model_discriminator,
                loss_func=loss_func_discriminator,
                optimizer=optimizer_discriminator,
                epochs=config["discriminator"]["pretrain_epochs"],
                inputs=all_inputs)
    avg_acc, avg_f1 = test_discrim(model=model_discriminator,
                loss_func=loss_func_discriminator,
                inputs=test_inputs)

    print(f"\nResults of {detector_version}: ========= \nAverage accuracy: {avg_acc:.4f}\nAverage f1-score: {avg_f1:.4f}")