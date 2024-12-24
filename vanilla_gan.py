from datasets import load_dataset
import pandas as pd
import torch
from torch import nn
import json
from models.discriminator import BotRGCN, train_discrim, test_discrim
from Dataset import Twibot22
from data.raw_data.preprocess import tweets_embedding
from models.feature_extrator import Feature_extractor
import random
from tqdm import tqdm
import os
import re
import argparse
import numpy as np

os.environ["WANDB_DISABLED"]="true"
# os.environ["WANDB_DISABLED"]="true"

PRECISION_MAPPING = {16: torch.float16, 32: torch.float32, 64: torch.float64}
COMM = 2
processed_data_folder=f"/home/fanqi/llm_simulation/data/processed_data/community_{COMM}/"
bot_processed_data = f"/home/fanqi/llm_simulation/data/processed_data/community_{COMM}/"
raw_data_folder = f"/home/fanqi/llm_simulation/data/raw_data/community_{COMM}/"
detector_folder = "/home/fanqi/llm_simulation/models/Detector/"
device = "cuda:0"
with open("/home/fanqi/llm_simulation/config.json", "r") as config_file:
        config = json.load(config_file)

tweets_df = pd.read_json(raw_data_folder + "tweet.json")
with open(processed_data_folder + "id_tweet.json", "r", encoding="utf-8") as f:
    human_tweet = json.load(f)

users_df = pd.read_json(raw_data_folder + "user_summary.json")
labels_df = pd.read_csv(raw_data_folder + "label.csv")
user_idx=labels_df['id']
uid_index={uid:index for index,uid in enumerate(user_idx.values)}
bot_ids = labels_df[labels_df['label'] == 'bot']['id'].tolist()
bot_num = len(bot_ids)
bot_indices = [uid_index[key] for key in bot_ids if key in uid_index]
human_ids = labels_df[labels_df['label'] == 'human']['id'].tolist()
human_indices = [uid_index[key] for key in human_ids if key in uid_index]
user_ids = user_idx.values
indexed_human_ids = list(enumerate(human_ids))
batch_size = config["generator_training"]["batch_size"]
training_epoch = config["generator_training"]["epoch"]

bot_dataset = Twibot22(root=bot_processed_data, device=device,process=False,save=False)

des_tensor1,tweets_tensor1,num_prop1,category_prop1,edge_index1,edge_type1,labels1,train_idx1,val_idx1,test_idx1=bot_dataset.dataloader(tweets_path="tweets_tensor.pt")

human_tweets_embeddings = tweets_tensor1[human_indices]
# inputs=[des_tensor, tweets_tensor, num_prop, category_prop, edge_index, edge_type, labels, train_idx, val_idx, test_idx]
bot_inputs = [des_tensor1, tweets_tensor1, num_prop1, category_prop1, edge_index1, edge_type1, labels1, train_idx1, val_idx1, test_idx1]

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=768, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=bot_num, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
opt = parser.parse_args()
print(opt)

img_shape = (1, opt.img_size)

cuda = True if torch.cuda.is_available() else False


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *img_shape).squeeze()
        return img

# Loss function
adversarial_loss = torch.nn.CrossEntropyLoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = BotRGCN(cat_prop_size=config["discriminator"]["cat_prop_size"],
                            embedding_dimension=config['discriminator']["embedding_dimension"]).to(device)

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----------
#  Training
# ----------

for epoch in range(opt.n_epochs):

    # -----------------
    #  Train Generator
    # -----------------

    optimizer_G.zero_grad()

    # Sample noise as generator input
    z = torch.tensor(np.random.normal(0, 1, (bot_num, opt.latent_dim)), 
                     dtype=torch.float).to(device)

    # Generate all bots' tweets tensor
    gen_bot_tweets_embedding = generator(z)

    new_tweets_tensor = tweets_tensor1.detach()
    new_tweets_tensor[bot_indices] = gen_bot_tweets_embedding
    output = discriminator(des_tensor1,
                                    new_tweets_tensor,
                                    num_prop1,
                                    category_prop1,
                                    edge_index1,
                                    edge_type1)
    generator_loss = adversarial_loss(output[bot_indices], torch.zeros(bot_num, dtype=torch.long, device=device))
    
    generator_loss.backward()
    optimizer_G.step()

    # ---------------------
    #  Train Discriminator
    # ---------------------

    optimizer_D.zero_grad()
    gen_bot_tweets_embedding = generator(z)

    new_tweets_tensor = tweets_tensor1.detach()
    new_tweets_tensor[bot_indices] = gen_bot_tweets_embedding
    output = discriminator(des_tensor1,
                                    new_tweets_tensor,
                                    num_prop1,
                                    category_prop1,
                                    edge_index1,
                                    edge_type1)
    # Measure discriminator's ability to classify real from generated samples
    discriminator_loss = adversarial_loss(output, labels1)

    discriminator_loss.backward()
    optimizer_D.step()

    print(
        "[Epoch %d/%d] [D loss: %f] [G loss: %f]"
        % (epoch, opt.n_epochs, discriminator_loss.item(), generator_loss.item())
        )