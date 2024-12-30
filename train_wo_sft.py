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

os.environ["WANDB_DISABLED"]="true"
# os.environ["WANDB_DISABLED"]="true"

PRECISION_MAPPING = {16: torch.float16, 32: torch.float32, 64: torch.float64}
COMM = 5
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
model_path = "NousResearch/Llama-2-7b-chat-hf"

dpo_dataset_path = "/home/fanqi/llm_simulation/data/dpo_data"
model_discriminator = BotRGCN(cat_prop_size=config["discriminator"]["cat_prop_size"],
                            embedding_dimension=config['discriminator']["embedding_dimension"]).to(other_device)
optimizer_discriminator = torch.optim.AdamW(model_discriminator.parameters(),
                        lr=config["discriminator"]["lr"],weight_decay=config["discriminator"]["weight_decay"])
loss_func_discriminator = nn.CrossEntropyLoss()

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
                # if re.search(r'Tweet \d+:', each_response):
                #     segments = re.split(r'Tweet \d+:', each_response.strip().replace('\n', ''))
                #     segments = [segments[i].strip() for i in range(1, len(segments))]
                # elif re.search(r'\d+\.', each_response):
                #     segments = re.split(r'\d+\.', each_response.strip().replace('\n', ''))
                #     segments = [segments[i].strip() for i in range(1, len(segments))]
                # elif re.search(r'\d+:', each_response):
                #     segments = re.split(r'\d+:', each_response.strip().replace('\n', ''))
                #     segments = [segments[i].strip() for i in range(1, len(segments))]
                # elif re.search(r'\d+\)', each_response):
                #     segments = re.split(r'\d+\)', each_response.strip().replace('\n', ''))
                #     segments = [segments[i].strip() for i in range(1, len(segments))]
                # elif re.search(r'\n\n', each_response):
                #     segments = re.split(r'\n\n', each_response.strip())
                #     segments = [segments[i].strip() for i in range(1, len(segments))]
                # else:
                #     segments = [each_response.strip()]
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
    # merge_peft_adapters(adapter_dir=f"/home/fanqi/llm_simulation/models/DPO/community_{COMM}/result_{0}", output_path=f"/home/fanqi/llm_simulation/models/DPO/community_{COMM}/merged_ckp_{0}")
    # exit()
    # prompt1 = "[INST]You are using the social media Twitter. Here is the discription about you: **Character Description: RocDev**  \nRocDev is an engaged tech enthusiast based in Rochester, NY, with a passion for fostering community within the developer space. Since joining the platform in late 2017, they’ve cultivated a network of 97 followers while actively following 119 accounts, indicating openness to learning and collaboration. With 135 tweets, their content reflects a strong interest in technology discussions, developer relations, and local events. They share insightful quotes from industry leaders, promote local tech meetups, and express encouragement towards peers, showcasing their supportive nature. RocDev has a curious and community-oriented spirit, often amplifying others’ voices and promoting skill-sharing through various tech topics, positioning them as an accessible resource for fellow developers..\nAdditionally, you also know information about several of your neighbors in the social network (i.e., users with whom you have a following or followed-by relationship): {'Neighbor 1': \"Aniela Wolkonowski, a New York State resident, embodies a blend of intellectual curiosity and whimsical humor, reflected in her playful engagement with philosophical themes. With 20 followers and a vast following of 79, she seems to foster a close-knit community of like-minded thinkers. Her modest tweet count of 316 suggests an emphasis on quality over quantity, indicating thoughtful contributions rather than mere chatter. Aniela likely enjoys exploring abstract concepts, inviting her audience to ponder life's complexities through clever wit and philosophical references.\", 'Neighbor 2': 'We Pivot is a community-driven initiative located in Rochester, NY, founded by the visionary @kr1573n. Passionate about inclusivity, it strives to empower marginalized individuals to enter the local tech scene. With a modest yet devoted following, We Pivot embodies optimism and resilience, focusing on mentorship and support. The account reflects a vibrant yet emerging voice, with intent to grow its impact and leadership team. Their activity suggests a commitment to dialogue and advocacy, aiming to foster connections and amplify underrepresented voices in technology.', 'Neighbor 3': \"Dan Schneiderman is an enthusiastic community engagement specialist based in Rochester, NY. As Eclipse Partnership Manager at the Rochester Museum and Science Center, he's dedicated to preparing the city for the upcoming 2024 Total Solar Eclipse, showcasing his passion for science and public engagement. His role as Community Manager at Maker Faire Rochester highlights his commitment to fostering creativity and innovation within the local community. Dan's Twitter activity is characterized by his proactive approach to networking, sharing knowledge, and promoting events, reflected in his substantial follower count and high tweet volume. He embodies a collaborative spirit, always eager to connect with others and share his insights while encouraging participation in community initiatives.\", 'Neighbor 4': \"**Character Description: NextCorps**  \\nNextCorps is a vibrant and innovative force in Rochester, NY, serving as the city's largest startup incubator. With a commitment to fostering entrepreneurship, they host a variety of engaging events, cultivate community connections, and provide invaluable mentorship. Their personality is energetic, supportive, and forward-thinking, reflecting a strong dedication to both local businesses and emerging technologies.  \\n\\n**Twitter Activity Summary:**  \\nActive and community-oriented, NextCorps tweets about startup success stories, innovative projects, and local events, all while promoting collaboration among tech enthusiasts and entrepreneurs. They engage frequently with followers, share valuable resources, and highlight opportunities for growth in the Rochester startup ecosystem.\"}\nNow, based on the above information, please generate several tweets. The topics are unrestricted, but they should fully showcase your personal characteristics and integrate into the online community.[/INST] "
    # prompt2 = "[INST]You are using the social media Twitter. Here is the discription about you: **Character Description: BWRiverkeeper**\n\nBWRiverkeeper is a passionate environmental advocate based in Alabama, dedicated to the protection and restoration of the Black Warrior River and its tributaries. As a long-standing nonprofit member of the Waterkeeper Alliance, they embody a commitment to clean water and community engagement. Their activity on Twitter reflects a proactive approach, showcasing a blend of organization, information sharing, and volunteer coordination. With over 31,000 tweets, their feeds are filled with retweets of environmental news, community events, and calls to action, illustrating a collaborative spirit and a determination to rally support for local initiatives. BWRiverkeeper is seen as an inspirational leader, fostering a sense of collective responsibility toward water conservation in Alabama, while emphasizing the importance of education and community involvement..\nAdditionally, you also know information about several of your neighbors in the social network (i.e., users with whom you have a following or followed-by relationship): {'Neighbor 1': \"**Character Description: Robert Gardner**\\n\\nRobert Gardner is an eco-conscious individual deeply engaged in environmental and social justice issues. His tweets showcase a passionate commitment to activism, often amplifying voices and movements that advocate for climate action and systemic change. With a network of 41 followers and 308 accounts he follows, he serves as a connector within progressive circles. His retweeting behavior highlights his support for various causes, suggesting he values community and collective efforts. Robert's tone indicates a blend of determination and urgency, reflecting a personality that is empathetic, socially aware, and motivated to challenge the status quo. His limited tweet count suggests a thoughtful approach to sharing content, favoring quality over quantity in his online presence.\", 'Neighbor 2': '**Byron Short Jr** is a dedicated coalminer, husband, and father, embodying the hardworking spirit of his profession. His tweets reveal a light-hearted personality with a penchant for humor and pop culture, especially in gaming and wrestling. Byron engages with various communities, showing loyalty and enthusiasm for franchises like Mass Effect, voicing both excitement and constructive criticism. Despite his limited followers, he actively participates in conversations, sharing his passions and interests. His tweets reflect a blend of family-oriented values and a playful, fun-loving nature, showcasing a man grounded in his roots yet open to the digital world around him.', 'Neighbor 3': 'Maniruzzaman is a detail-oriented and creative individual, blending the analytical mind of a CPA marketer with the artistic flair of a web designer and developer. His professional background suggests strong problem-solving skills and a meticulous approach to projects. With a modest following, he appears selective in social interactions, showcasing a preference for quality connections over quantity. His minimal Twitter activity indicates a thoughtful and possibly introverted personality, likely taking time to craft meaningful contributions when he engages.'}\nNow, based on the above information, please generate several tweets. The topics are unrestricted, but they should fully showcase your personal characteristics and integrate into the online community.[/INST] "
    # batch_prompt = [prompt2, prompt1]
    # test_generate_data(batch_prompt, model_path=model_path, device = "cuda:2")
    # exit()
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
        dpo_replaced_tweet_path = processed_data_folder + f"id_tweet_dpo_{epoch}_wo_sft.json"
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
        updated_tweets_tensor_path = processed_data_folder + f"tweets_tensor_dpo_{epoch}_wo_sft.pt"
        if not os.path.exists(updated_tweets_tensor_path):
            tweets_embedding(each_user_tweets_path=dpo_replaced_tweet_path, 
                             output_path=updated_tweets_tensor_path,
                             community=COMM, dpo_epoch=epoch, device=other_device)
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
        dpo_data_path = f"{dpo_dataset_path}/community_{COMM}/dpo_dataset_{epoch}_wo_sft.jsonl"
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
        dpo_model_path = f"/home/fanqi/llm_simulation/models/DPO/community_{COMM}/merged_ckp_{epoch}_wo_sft"
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
            DPO_trainer.save_model(f"/home/fanqi/llm_simulation/models/DPO/community_{COMM}/result_{epoch}_wo_sft")
            # DPO_trainer.model.save_pretrained(f"/home/fanqi/llm_simulation/models/DPO/DPO_ckp_{epoch}")
            # model_generator.tokenizer.save_pretrained(f"/home/fanqi/llm_simulation/models/DPO/DPO_ckp_{epoch}")

            # update generator
            merge_peft_adapters(adapter_dir=f"/home/fanqi/llm_simulation/models/DPO/community_{COMM}/result_{epoch}_wo_sft", output_path=dpo_model_path)
        model_path = dpo_model_path
        # exit()
