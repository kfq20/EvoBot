'''
The detector is trained on SFT generator dataset, no updated.
'''

from datasets import load_dataset
from trl import DPOConfig, DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
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
from data.raw_data.preprocess import extract_tweets_1, tweets_embedding
from models.feature_extrator import Feature_extractor
from utils.prompt_tempate import find_user_neighbors, llama_prompt
from utils.merge_peft_adapters import merge_peft_adapters
from utils.rewrite_bot_tweet import rewrite_all_bot_tweets, get_free_gpus
from torch.multiprocessing import Process, Queue, set_start_method, Manager
import random
from tqdm import tqdm
import os
import re
os.environ["WANDB_DISABLED"]="true"
# os.environ["WANDB_DISABLED"]="true"

PRECISION_MAPPING = {16: torch.float16, 32: torch.float32, 64: torch.float64}
COMM = 2
processed_data_folder=f"/home/fanqi/llm_simulation/data/processed_data/community_{COMM}/"
bot_processed_data = f"/home/fanqi/llm_simulation/data/processed_data/community_{COMM}/"
raw_data_folder = f"/home/fanqi/llm_simulation/data/raw_data/community_{COMM}/"
detector_folder = "/home/fanqi/llm_simulation/models/Detector/"
generator_device = 'cuda:1'
other_device = "cuda:0"

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
human_ids = labels_df[labels_df['label'] == 'human']['id'].tolist()
indexed_human_ids = list(enumerate(human_ids))
DPO_config = DPOConfig(**config["DPO_trainer"], fp16=True, gradient_checkpointing=True)
peft_config = LoraConfig(**config["peft"])
feature_extract=pipeline('feature-extraction',model='roberta-base', device=other_device, tokenizer='roberta-base', truncation=True,padding='max_length', add_special_tokens = True, max_length=512)
feature_extract.tokenizer.model_max_length = 512
dpo_dataset_size = config["DPO_dataset_size"]
batch_size = config["generator_training"]["batch_size"]
training_epoch = config["generator_training"]["epoch"]

dataset=Twibot22(root=processed_data_folder,device=other_device,process=False,save=False)
bot_dataset = Twibot22(root=bot_processed_data, device=other_device,process=False,save=False)

des_tensor1,tweets_tensor1,num_prop1,category_prop1,edge_index1,edge_type1,labels1,train_idx1,val_idx1,test_idx1=bot_dataset.dataloader(tweets_path="tweets_tensor.pt")
des_tensor,tweets_tensor,num_prop,category_prop,edge_index,edge_type,labels,train_idx,val_idx,test_idx=dataset.dataloader(tweets_path="tweets_tensor.pt")

inputs=[des_tensor, tweets_tensor, num_prop, category_prop, edge_index, edge_type, labels, train_idx, val_idx, test_idx]
bot_inputs = [des_tensor1, tweets_tensor1, num_prop1, category_prop1, edge_index1, edge_type1, labels1, train_idx1, val_idx1, test_idx1]
model_path = f"/home/fanqi/llm_simulation/models/SFT/sft_merged_ckp"
dpo_dataset_path = "/home/fanqi/llm_simulation/data/dpo_data"
model_discriminator = BotRGCN(cat_prop_size=config["discriminator"]["cat_prop_size"],
                            embedding_dimension=config['discriminator']["embedding_dimension"]).to(other_device)
optimizer_discriminator = torch.optim.AdamW(model_discriminator.parameters(),
                        lr=config["discriminator"]["lr"],weight_decay=config["discriminator"]["weight_decay"])
loss_func_discriminator = nn.CrossEntropyLoss()
vanilla_LLM_replace_human_tweet_path = raw_data_folder + "vanilla_LLM_replace_human.json"
with open(vanilla_LLM_replace_human_tweet_path, "r", encoding="utf-8") as f:
    data = json.load(f)
# Access the tweets and human_ids
replaced_human_ids = data.get("replaced_human_ids", [])
replaced_human_indices = torch.tensor(data.get("replaced_human_indices", []), device=other_device)
inputs[6][replaced_human_indices] = 1

'''raw data pretraining'''
# print(f"\n=== Train detector on human data ===\n")
# train_discrim(model=model_discriminator,
#                     loss_func=loss_func_discriminator,
#                     optimizer=optimizer_discriminator,
#                     epochs=config["discriminator"]["pretrain_epochs"],
#                     inputs=inputs)
# test_discrim(model=model_discriminator,
#                     loss_func=loss_func_discriminator,
#                     inputs=inputs)

def build_DPO_data(prompts, response_1, response_2, score_1, score_2):
    '''score: [batch, 2], where the first dim is the logits of human label'''
    filtered_data = []
    for i, prompt in enumerate(prompts):
        filtered_data.append({
            "prompt": prompt,
            "chosen": response_1[i] if score_1[i][0] >= score_2[i][0] else response_2[i],
            "rejected": response_2[i] if score_2[i][0] <= score_1[i][0] else response_1[i]
        })
    return filtered_data

def test_generate_data(batch_prompt, model_path, device):
    model_args = {
        "torch_dtype": PRECISION_MAPPING[config["llm_model"]["precision_bits"]],
        "device_map": {"": device}
    }
    model_generator = Generator(model_path, **model_args)
    model_generator.model = get_peft_model(model_generator.model, peft_config)
    batch_response_1 = model_generator.generate_text(batch_prompt, max_length=2048, temperature=1.0, do_sample=True, repetition_penalty=1.3)
    processed_outputs = []
    for output in batch_response_1:
        response_output = re.sub(r'\[INST\].*?\[/INST\]', '', output, flags=re.DOTALL).strip()
        processed_outputs.append(response_output)
    print(f"=== prompt ===\n{batch_prompt}\n")
    print(f"=== response ===\n{processed_outputs}")

def generate_data(gpu_id, queue, samples, model_path, model_discriminator):
    torch.cuda.set_device(gpu_id)
    device = f"cuda:{gpu_id}"
    model_args = {
        "torch_dtype": PRECISION_MAPPING[config["llm_model"]["precision_bits"]],
        "device_map": {"": device}
    }
    model_generator = Generator(model_path, **model_args)
    model_generator.model = get_peft_model(model_generator.model, peft_config)

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
            batch_response_1 = model_generator.generate_text(batch_prompt, max_length=2048, temperature=1, do_sample=True, repetition_penalty=1.3)
            batch_response_2 = model_generator.generate_text(batch_prompt, max_length=2048, temperature=1, do_sample=True, repetition_penalty=1.3)
            response_tensor = []
            for each_response in batch_response_1 + batch_response_2:
                if re.search(r'Tweet \d+:', each_response):
                    segments = re.split(r'Tweet \d+:', each_response.strip().replace('\n', ''))
                    segments = [segments[i].strip() for i in range(1, len(segments))]
                elif re.search(r'\d+\.', each_response):
                    segments = re.split(r'\d+\.', each_response.strip().replace('\n', ''))
                    segments = [segments[i].strip() for i in range(1, len(segments))]
                elif re.search(r'\d+:', each_response):
                    segments = re.split(r'\d+:', each_response.strip().replace('\n', ''))
                    segments = [segments[i].strip() for i in range(1, len(segments))]
                elif re.search(r'\d+\)', each_response):
                    segments = re.split(r'\d+\)', each_response.strip().replace('\n', ''))
                    segments = [segments[i].strip() for i in range(1, len(segments))]
                elif re.search(r'\n\n', each_response):
                    segments = re.split(r'\n\n', each_response.strip())
                    segments = [segments[i].strip() for i in range(1, len(segments))]
                else:
                    segments = [each_response.strip()]

                all_tweet_tensor=torch.tensor(feature_extract(segments, padding=True, max_length=512, truncation=True, batch_size=len(segments)))
                total_word_tensor = torch.mean(all_tweet_tensor, dim=2)
                total_each_person_tweets = torch.mean(total_word_tensor, dim=0)
                response_tensor.append(total_each_person_tweets)

            response_tensor = torch.cat(response_tensor, dim=0).to(other_device)
            # response_tensor = feature_extractor.feature_extract(batch_response_1+batch_response_2).to(other_device)
        
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
    # prompt1 = "[INST]You are using the social media Twitter. Here is the discription about you: Walter Ricciardi is a distinguished Public Health professor based in Rome, Italy, with an extensive background in health policy and research. As a prominent figure in the global health community, he serves as the Scientific Director at Maugeri and chairs the Mission Board for Cancer at the European Commission. His leadership roles, including past President of ISTISUPAN and current President of the World Federation of Public Health Associations (WFPHA-FMASP), reflect his dedication to improving public health on an international scale. \n\nWith a significant following on Twitter, Walter actively engages in discussions about pandemic response, vaccination advocacy, and health policy reform, showcasing a strong commitment to science-based public health initiatives. His tweets reveal a proactive stance on critical health issues, often amplifying calls for vaccinations and global cooperation in health emergencies. His presence on social media reflects a blend of expertise, advocacy, and a collaborative spirit aimed at enhancing the health of populations worldwide. \n\nPersonality traits: Knowledgeable, passionate, influential, collaborative, and proactive..\nAdditionally, you also know information about several of your neighbors in the social network (i.e., users with whom you have a following or followed-by relationship): {'Neighbor 1': 'Tit Albreht, a dedicated health services researcher based in Ljubljana, Slovenia, possesses a deep commitment to advancing health equity and international health policy. His extensive engagement on Twitter reveals a collaborative spirit, as he frequently shares insights from various experts and participates in discussions at significant health forums. Tit demonstrates a strong advocacy for vaccination, public health awareness, and the importance of inclusive healthcare initiatives. His work is characterized by a thoughtful and analytical approach, and he values evidence-based policies. With a follower count exceeding 1,000, he actively connects with a broad network of professionals in the health sector, highlighting his influential presence in health discourse.'}\nNow, based on the above information, please generate several tweets. The topics are unrestricted, but they should fully showcase your personal characteristics and integrate into the online community.[/INST]"
    # prompt2 = "[INST]You are using the social media Twitter. Here is the discription about you: Ody is a passionate and expressive individual, evident from their active engagement on Twitter. With a follower count of 33 and a following of 62, they navigate social media with an enthusiastic focus on celebrity culture, particularly around the figure of Can Yaman. Ody's retweets reflect a supportive and community-oriented mindset, celebrating successes and sharing news related to their interests. Their tweets also reveal a tendency to defend and uplift others, showcasing a blend of admiration and assertiveness. Overall, Ody exudes warmth and a strong sense of loyalty, embedded in a vibrant online presence..\nAdditionally, you also know information about several of your neighbors in the social network (i.e., users with whom you have a following or followed-by relationship): {'Neighbor 1': 'Alessia Marcuzzi is a spirited and compassionate individual, embodying a blend of courage and kindness. Her vibrant personality shines through her Twitter activity, where she engages with a multitude of topics, showcasing her support for others and a deep concern for pressing societal issues. With a substantial following, Alessia often shares heartfelt messages and offers encouragement, whether celebrating accomplishments or advocating for those in need. Her online presence reflects a strong connection to her followers, marked by her willingness to involve herself in diverse conversations while radiating authenticity and warmth.', 'Neighbor 2': \"Sebla \u00d6zveren is a dynamic producer and DJ based in \u0130stanbul, originally hailing from \u0130zmir. With a creative spirit fueled by a passion for music, Sebla thrives in the vibrant atmosphere of Turkey's cultural scene. Known for a keen sense of collaboration and innovation, she engages her audience through her notable presence at the Turkish Radio and Television Corporation. Her Twitter activity reflects a proactive approach to connecting with followers, showcasing a diverse array of interests and insights, underscored by a consistent engagement with the community. With nearly 8,325 tweets, her voice resonates within the music and media landscape, making her a relatable figure among her 1,886 followers.\", 'Neighbor 3': \"**Character Description: Daniela**  \\nDaniela is a passionate and articulate individual deeply allergic to rudeness and ignorance. Her Italian roots shine through her vibrant personality, and she carries a playful wit paired with an unwavering commitment to civility. With an impressive Twitter activity, her engagement with over 33,000 tweets reflects her eagerness to connect and share ideas. Daniela values meaningful interactions and is likely to champion thoughtful dialogue while playfully calling out those who embrace a lack of respect as a lifestyle choice. She's a true advocate for kindness and understanding in the digital space.\", 'Neighbor 4': 'Ilaria is a vibrant and artistic soul based in Firenze, reflecting her passion for creativity and life. With an affectionate touch, symbolized by her love for Margherita flowers, she brings warmth and joy to her interactions. Her Twitter activity showcases a lively engagement with a diverse community, evidenced by her substantial tweet count. Ilaria exhibits traits of sociability, expressiveness, and enthusiasm, drawing in her followers with a mix of charm and insightful creativity.', 'Neighbor 5': \"Diletta Leotta is a charismatic and engaging sports presenter known for her work with DAZN and Radio105. With a robust following of over 211,000, she exudes confidence and charm, making her a beloved figure in sports broadcasting. Her Twitter activity reflects her vibrant personality, sharing insights, updates, and interactions that resonate with her audience. Leotta's presence suggests she is dynamic, approachable, and passionate about her field.\"}\nNow, based on the above information, please generate several tweets. The topics are unrestricted, but they should fully showcase your personal characteristics and integrate into the online community.[/INST]"
    # batch_prompt = [prompt1, prompt2]
    # test_generate_data(batch_prompt, model_path=model_path, device = "cuda:2")
    # exit()
    
    '''Start main training loop'''

    print("\n=== Start main training loop! ===\n")
    print(f"\n=== Current model {model_path} ===\n")
    set_start_method('spawn', force=True) 
    free_gpus = get_free_gpus()
    print(f"Free GPUs found: {free_gpus}")
    num_gpus = len(free_gpus)
    updated_tweets_tensor=torch.load(processed_data_folder + f"tweets_dpo_0_tensor.pt").to(other_device)
    bot_inputs[1] = updated_tweets_tensor
    '''train detector'''
    print(f"\n=== detector training: 0 ===\n")
    model_discriminator = BotRGCN(cat_prop_size=config["discriminator"]["cat_prop_size"],
                        embedding_dimension=config['discriminator']["embedding_dimension"]).to(other_device)
    optimizer_discriminator = torch.optim.AdamW(model_discriminator.parameters(),
                    lr=config["discriminator"]["lr"],weight_decay=config["discriminator"]["weight_decay"])
    train_discrim(model=model_discriminator,
            loss_func=loss_func_discriminator,
            optimizer=optimizer_discriminator,
            epochs=config["discriminator"]["pretrain_epochs"],
            inputs=bot_inputs)
    test_discrim(model=model_discriminator,
                loss_func=loss_func_discriminator,
                inputs=bot_inputs)
    exit()
    for epoch in range(0, training_epoch):
        print(f"=== epoch: {epoch} ===")

        '''use the sft model or dpo model to rewrite the replaced human tweet'''
        dpo_replaced_human_tweet_path = processed_data_folder + f"id_tweet_dpo_0.json"
        if not os.path.exists(dpo_replaced_human_tweet_path):
            bot_ids_per_gpu = len(replaced_human_ids) // num_gpus 
            manager = Manager()
            queue = manager.Queue()
            processes = []
            for i, gpu_id in enumerate(free_gpus):
                start_idx = i * bot_ids_per_gpu
                end_idx = start_idx + bot_ids_per_gpu
                if i != num_gpus - 1:
                    bot_ids_gpu = replaced_human_ids[start_idx:end_idx]
                else:
                    bot_ids_gpu = replaced_human_ids[start_idx:]
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
            replaced_human_tweets = []
            for human_index in human_tweet:
                if int(human_index) in replaced_human_indices: # replaced human
                    replaced_human_tweets.append({human_index: replaced_tweet_dict[human_ids[int(human_index)]]})
                else:
                    replaced_human_tweets.append({human_index: human_tweet[human_index]})

            id_replaced_tweets = {k: v for d in replaced_human_tweets for k, v in d.items()}
            with open(processed_data_folder + f"id_tweet_dpo_{epoch}.json", "w", encoding="utf-8") as f:
                json.dump(id_replaced_tweets, f, indent=4, ensure_ascii=False)
        
        '''update tweets tensor'''
        updated_tweets_tensor_path = processed_data_folder + f"tweets_dpo_{epoch}_tensor.pt"
        if not os.path.exists(updated_tweets_tensor_path):
            tweets_embedding(each_user_tweets_path=processed_data_folder + f"id_tweet_dpo_{epoch}.json", 
                             output_path=updated_tweets_tensor_path,
                             community=COMM, dpo_epoch=epoch, device=other_device)
        
        # exit()
        
        '''sample dpo dataset'''
        dpo_data_path = f"{dpo_dataset_path}/community_{COMM}/wo_adv_dpo_dataset_{epoch}.jsonl"
        if not os.path.exists(dpo_data_path):
            samples_per_gpu = int(dpo_dataset_size/batch_size) // num_gpus
            manager = Manager()
            queue = manager.Queue()
            processes = []

            for i, gpu_id in enumerate(free_gpus):
                p = Process(target=generate_data, args=(gpu_id, queue, samples_per_gpu, model_path, model_discriminator))
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
        dpo_model_path = f"/home/fanqi/llm_simulation/models/DPO/wo_adv_merged_ckp_{epoch}"
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
            DPO_trainer.save_model(f"/home/fanqi/llm_simulation/models/DPO/result_{epoch}")
            # DPO_trainer.model.save_pretrained(f"/home/fanqi/llm_simulation/models/DPO/DPO_ckp_{epoch}")
            # model_generator.tokenizer.save_pretrained(f"/home/fanqi/llm_simulation/models/DPO/DPO_ckp_{epoch}")

            # update generator
            merge_peft_adapters(adapter_dir=f"/home/fanqi/llm_simulation/models/DPO/result_{epoch}", output_path=f"/home/fanqi/llm_simulation/models/DPO/merged_ckp_{epoch}")
        model_path = dpo_model_path
        # exit()

