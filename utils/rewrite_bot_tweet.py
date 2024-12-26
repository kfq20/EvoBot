from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.distributed as dist
import pandas as pd
import torch

import json

import random
from tqdm import tqdm
import subprocess
import re
import os

def test_model_output(model_path, prompt, device):
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).to(device)
    print("=== successfully implemented llm! ===")
    tokenizer = AutoTokenizer.from_pretrained(model_path, model_max_length=2048)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model.config.pad_token_id = tokenizer.pad_token_id
    inputs = tokenizer(prompt, padding=True, truncation=True, return_tensors="pt", max_length=1024).to(device)
    output = model.generate(**inputs, max_new_tokens=1024, temperature=0.7, do_sample=True, repetition_penalty=1.3, pad_token_id = tokenizer.pad_token_id)
    output_text = tokenizer.batch_decode(output, skip_special_tokens=True)
    processed_outputs = []
    for output in output_text:
        response_output = re.sub(r'\[INST\].*?\[/INST\]', '', output, flags=re.DOTALL).strip()
        processed_outputs.append(response_output)
    print(f"=== prompt ===\n{prompt}\n")
    for output in processed_outputs:
        print(f"=== response ===\n{output}")
    

def rewrite_all_bot_tweets(gpu_id, model_path, bot_ids, community, queue):
    url_pattern1 = re.compile(r'https?://\S+')
    url_pattern2 = re.compile(r'http?://\S+')

    from .prompt_tempate import find_user_neighbors, llama_prompt
    torch.cuda.set_device(gpu_id)
    device = f"cuda:{gpu_id}"
    folder_path = f"/home/fanqi/llm_simulation/data/raw_data/community_{community}/"
    users_df = pd.read_json(folder_path + "user_summary.json")
    labels_df = pd.read_csv(folder_path + "label.csv")
    user_idx=labels_df['id']

    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).to(device)
    print("=== successfully implemented llm! ===")
    tokenizer = AutoTokenizer.from_pretrained(model_path, model_max_length=2048)
    model.config.pad_token_id = tokenizer.pad_token_id
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    batch_size = 4
    new_tweets = []
    for i in tqdm(range(int(len(bot_ids)//batch_size))):
        if i != int(len(bot_ids)//batch_size)-1:
            sampled_bot_ids = bot_ids[i*batch_size:(i+1)*batch_size]
        else:
            sampled_bot_ids = bot_ids[i*batch_size:]
        batch_prompt = []
        for bot_id in sampled_bot_ids:
            neighbor_ids = find_user_neighbors(folder_path + "edge.csv", bot_id, 5)
            neighbor_infos = []
            for neighbor_id in neighbor_ids:
                try:
                    neighbor_infos.append(users_df[users_df['id'] == neighbor_id].iloc[0])
                except:
                    pass
            user_info = users_df[users_df['id'] == bot_id].iloc[0]
            prompt = llama_prompt(user_info, neighbor_infos)
            # prompt = "[INST] hello, today is a good day. [/INST]"
            batch_prompt.append(prompt)
        with torch.no_grad():
            inputs = tokenizer(batch_prompt, padding=True, truncation=True, return_tensors="pt", max_length=2048).to(device)
            output = model.generate(**inputs, max_new_tokens=512, temperature=0.7, do_sample=True, repetition_penalty=1.3, pad_token_id = tokenizer.pad_token_id)
            output_text = tokenizer.batch_decode(output, skip_special_tokens=True)
            processed_outputs = []
            for output in output_text:
                response_output = re.sub(r'\[INST\].*?\[/INST\]', '', output, flags=re.DOTALL).strip()
                processed_outputs.append(response_output)
        
        for bot_id, text in zip(sampled_bot_ids, processed_outputs):
            # if re.search(r'Tweet \d+:', text):
            #     segments = re.split(r'Tweet \d+:', text.strip().replace('\n', ''))
            #     segments = [segments[i].strip() for i in range(1, len(segments))]
            # elif re.search(r'Tweet #\d', text):
            #     segments = re.split(r'Tweet \d+:', text.strip().replace('\n', ''))
            #     segments = [segments[i].strip() for i in range(1, len(segments))]
            # elif re.search(r'\d+\.', text):
            #     segments = re.split(r'\d+\.', text.strip().replace('\n', ''))
            #     segments = [segments[i].strip() for i in range(1, len(segments))]
            # elif re.search(r'\d+:', text):
            #     segments = re.split(r'\d+:', text.strip().replace('\n', ''))
            #     segments = [segments[i].strip() for i in range(1, len(segments))]
            # elif re.search(r'\d+\)', text):
            #     segments = re.split(r'\d+\)', text.strip().replace('\n', ''))
            #     segments = [segments[i].strip() for i in range(1, len(segments))]
            if re.search(r'\n', text):
                segments = re.split(r'\n', text.strip())
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
                results = [text]
            new_tweets.append({
                'author_id': bot_id,
                'text':results
            })
    print(f"=== Process {gpu_id} done ===") 
    queue.put((gpu_id, new_tweets))
    # with open(f'/home/fanqi/llm_simulation/data/raw_data/community_{community}/llama_bot_tweet_{gpu_id}.json', 'w') as f:
    #     json.dump(new_tweets, f, ensure_ascii=False, indent=4)

import os
import subprocess

def get_free_gpus():
    """
    获取当前环境下（受 CUDA_VISIBLE_DEVICES 限制）的空闲 GPU 列表，返回逻辑 GPU 编号。
    """
    try:
        # 获取 CUDA_VISIBLE_DEVICES 环境变量
        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cuda_visible_devices is not None:
            # 解析为物理 GPU 列表
            visible_gpus = list(map(int, cuda_visible_devices.split(',')))
        else:
            # 未设置时，假定所有 GPU 可见
            result = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                encoding="utf-8",
            )
            visible_gpus = list(range(len(result.strip().split("\n"))))

        # 运行 nvidia-smi 并获取物理 GPU 的显存使用情况
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            encoding="utf-8",
        )
        memory_used = [int(x) for x in result.strip().split("\n")]

        # 筛选出空闲的物理 GPU
        free_physical_gpus = [idx for idx in visible_gpus if memory_used[idx] < 100]

        # 将物理编号映射为逻辑编号
        free_logical_gpus = [visible_gpus.index(phys_gpu) for phys_gpu in free_physical_gpus]

        return free_logical_gpus
    except Exception as e:
        print(f"Error while checking GPUs: {e}")
        return []

    
if __name__ == '__main__':
    test_model_output(
        model_path="/home/fanqi/llm_simulation/models/SFT/sft_merged_ckp",
                    #   model_path="NousResearch/Llama-2-7b-chat-hf",
                    #   model_path="/home/fanqi/llm_simulation/models/DPO/merged_ckp_0",

                    #   prompt=["[INST]You are using the social media Twitter. Here is the discription about you: Mark Soliman, MD, FACS, FASCRS, is an esteemed colorectal surgeon and Program Medical Director at AdventHealth's Digestive Health & Surgery Institute in Orlando, FL. He showcases a passion for advancing surgical techniques and education, evident in his frequent engagement with peers and commitment to patient care. Mark exhibits characteristics of collaboration, enthusiasm, and a sharp sense of humor, reflected in his retweets and interaction with fellow medical professionals. His extensive Twitter activity includes 5,306 tweets, where he shares insights on colorectal surgery, promotes medical advancements, and encourages community learning within the surgical field. As a social media presence, he has garnered a following of 4,547, indicating a strong connection to both medical professionals and the public..\nAdditionally, you also know information about several of your neighbors in the social network (i.e., users with whom you have a following or followed-by relationship): {'Neighbor 1': 'Heather Czujak is a warm-hearted and dedicated individual from Orlando, Florida, who balances her roles as a loving wife, a devoted mom to a son and a fur baby, and a successful medical marketer. As a two-time alumna of UCF, she embodies a strong Midwestern work ethic and a passion for fitness, particularly running. Her social media presence reflects her nurturing nature and commitment to an active lifestyle, though her limited tweet count suggests she prefers quality over quantity in her online interactions. With a small but engaged following, she shares snippets of her life that highlight her vibrant personality and deep connections with family and friends.'}\nNow, based on the above information, please generate several tweets. The topics are unrestricted, but they should fully showcase your personal characteristics and integrate into the online community.[/INST]",
                    #           "[INST]You are using the social media Twitter. Here is the discription about you: Ody is a passionate and expressive individual, evident from their active engagement on Twitter. With a follower count of 33 and a following of 62, they navigate social media with an enthusiastic focus on celebrity culture, particularly around the figure of Can Yaman. Ody's retweets reflect a supportive and community-oriented mindset, celebrating successes and sharing news related to their interests. Their tweets also reveal a tendency to defend and uplift others, showcasing a blend of admiration and assertiveness. Overall, Ody exudes warmth and a strong sense of loyalty, embedded in a vibrant online presence..\nAdditionally, you also know information about several of your neighbors in the social network (i.e., users with whom you have a following or followed-by relationship): {'Neighbor 1': 'Alessia Marcuzzi is a spirited and compassionate individual, embodying a blend of courage and kindness. Her vibrant personality shines through her Twitter activity, where she engages with a multitude of topics, showcasing her support for others and a deep concern for pressing societal issues. With a substantial following, Alessia often shares heartfelt messages and offers encouragement, whether celebrating accomplishments or advocating for those in need. Her online presence reflects a strong connection to her followers, marked by her willingness to involve herself in diverse conversations while radiating authenticity and warmth.', 'Neighbor 2': \"Sebla \u00d6zveren is a dynamic producer and DJ based in \u0130stanbul, originally hailing from \u0130zmir. With a creative spirit fueled by a passion for music, Sebla thrives in the vibrant atmosphere of Turkey's cultural scene. Known for a keen sense of collaboration and innovation, she engages her audience through her notable presence at the Turkish Radio and Television Corporation. Her Twitter activity reflects a proactive approach to connecting with followers, showcasing a diverse array of interests and insights, underscored by a consistent engagement with the community. With nearly 8,325 tweets, her voice resonates within the music and media landscape, making her a relatable figure among her 1,886 followers.\", 'Neighbor 3': \"**Character Description: Daniela**  \\nDaniela is a passionate and articulate individual deeply allergic to rudeness and ignorance. Her Italian roots shine through her vibrant personality, and she carries a playful wit paired with an unwavering commitment to civility. With an impressive Twitter activity, her engagement with over 33,000 tweets reflects her eagerness to connect and share ideas. Daniela values meaningful interactions and is likely to champion thoughtful dialogue while playfully calling out those who embrace a lack of respect as a lifestyle choice. She's a true advocate for kindness and understanding in the digital space.\", 'Neighbor 4': 'Ilaria is a vibrant and artistic soul based in Firenze, reflecting her passion for creativity and life. With an affectionate touch, symbolized by her love for Margherita flowers, she brings warmth and joy to her interactions. Her Twitter activity showcases a lively engagement with a diverse community, evidenced by her substantial tweet count. Ilaria exhibits traits of sociability, expressiveness, and enthusiasm, drawing in her followers with a mix of charm and insightful creativity.', 'Neighbor 5': \"Diletta Leotta is a charismatic and engaging sports presenter known for her work with DAZN and Radio105. With a robust following of over 211,000, she exudes confidence and charm, making her a beloved figure in sports broadcasting. Her Twitter activity reflects her vibrant personality, sharing insights, updates, and interactions that resonate with her audience. Leotta's presence suggests she is dynamic, approachable, and passionate about her field.\"}\nNow, based on the above information, please generate several tweets. The topics are unrestricted, but they should fully showcase your personal characteristics and integrate into the online community.[/INST]"],
                    prompt="[INST]You are using the social media Twitter. Here is the discription about you: Mark Soliman, MD, FACS, FASCRS, is an esteemed colorectal surgeon and Program Medical Director at AdventHealth's Digestive Health & Surgery Institute in Orlando, FL. He showcases a passion for advancing surgical techniques and education, evident in his frequent engagement with peers and commitment to patient care. Mark exhibits characteristics of collaboration, enthusiasm, and a sharp sense of humor, reflected in his retweets and interaction with fellow medical professionals. His extensive Twitter activity includes 5,306 tweets, where he shares insights on colorectal surgery, promotes medical advancements, and encourages community learning within the surgical field. As a social media presence, he has garnered a following of 4,547, indicating a strong connection to both medical professionals and the public..\nAdditionally, you also know information about several of your neighbors in the social network (i.e., users with whom you have a following or followed-by relationship): {'Neighbor 1': 'Heather Czujak is a warm-hearted and dedicated individual from Orlando, Florida, who balances her roles as a loving wife, a devoted mom to a son and a fur baby, and a successful medical marketer. As a two-time alumna of UCF, she embodies a strong Midwestern work ethic and a passion for fitness, particularly running. Her social media presence reflects her nurturing nature and commitment to an active lifestyle, though her limited tweet count suggests she prefers quality over quantity in her online interactions. With a small but engaged following, she shares snippets of her life that highlight her vibrant personality and deep connections with family and friends.'}\nNow, based on the above information, please generate several tweets. The topics are unrestricted, but they should fully showcase your personal characteristics and integrate into the online community.[/INST]",
                      device="cuda:4")