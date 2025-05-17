import json
import csv
from tqdm import tqdm
import random
import pandas as pd
import os
import re
from utils.prompt_tempate import llama_prompt

def extract_user_tweets(file_path, user_id, tweet_num):
    tweets = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            for tweet in data:
                if 'u' + str(tweet.get('author_id')) == user_id:
                    tweets.append(tweet)
                    if len(tweets) >= tweet_num:
                        break
        return tweets
    except Exception as e:
        print(f"An error occurred: {e}")
        return []
    
def extract_users_info(file_path, user_ids):
    users_info = []
    user_id_set = set(user_ids)
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            for user in data:
                if user.get('id') in user_id_set:
                    users_info.append(user)
                    if len(users_info) == len(user_id_set):
                        break
        return users_info
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

def find_user_neighbors(file_path, user_id, n):
    neighbors = set()  # Using a set to avoid duplicate neighbors
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                # Check if the relationship is either following or followers
                if row['relation'] in ['following', 'followers']:
                    # Add to neighbors if user_id is either source or target
                    if row['source_id'] == user_id:
                        neighbors.add(row['target_id'])         
                    elif row['target_id'] == user_id:
                        neighbors.add(row['source_id'])
                    
                    # Stop if we have reached the desired number of neighbors
                    if len(neighbors) >= n:
                        break
        return list(neighbors)[:n]  # Return exactly n neighbors
    except Exception as e:
        print(f"An error occurred: {e}")
        return []
    
def instruction_tune_instance(folder_path, processed_path, output_dir, data_items_num):
    each_user_tweets = json.load(open(f"{processed_path}/id_tweet.json", 'r'))
    users_df = pd.read_json(f"{folder_path}/user_summary.json")
    labels_df = pd.read_csv(f"{folder_path}/label.csv")
    user_idx=labels_df['id']
    uid_index={uid:index for index,uid in enumerate(user_idx.values)}

    human_ids = labels_df[labels_df['label'] == 'human']['id'].tolist()
    if len(human_ids) > data_items_num:
        human_ids = random.sample(human_ids, data_items_num)

    sft_data = []
        
    for human_id in tqdm(human_ids):
        neighbor_ids = find_user_neighbors(f"{folder_path}/edge.csv", human_id, 5)
        neighbor_infos = []

        user_info = users_df[users_df['id'] == human_id].iloc[0]
        for neighbor_id in neighbor_ids:
            try:
                neighbor_infos.append(users_df[users_df['id'] == neighbor_id].iloc[0])
            except:
                pass
        
        user_tweets = each_user_tweets[str(uid_index[human_id])]
        # user_tweets = tweets_df[tweets_df['author_id'] == int(human_id[1:])]['text'].tolist()
        if len(user_tweets) > 20:
            user_tweets = random.sample(user_tweets, 20)

        prompt = llama_prompt(user_info, neighbor_infos)

        response = ""
        for i, response_tweet in enumerate(user_tweets):
            if len(response_tweet) != 0:
                response += f"{i+1}. {response_tweet.strip()}\n"
        text = {"text": f"<s>{prompt} {response} </s>"}
        if response != "":
            sft_data.append(text)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(output_dir + "/sft_data.jsonl", "w", encoding="utf-8") as f:
        for record in sft_data:
            json.dump(record, f, ensure_ascii=False)
            f.write("\n")

if __name__ == "__main__":
    folder_path = "./data/raw_data/community_0"
    data_num = 1024
    instruction_tune_instance(folder_path, data_num)