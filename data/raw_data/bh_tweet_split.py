import json
import pandas as pd
import re

for i in [2]:
    # 读取 label.csv 文件
    folder_path = f'/home/fanqi/llm_simulation/data/raw_data/community_{i}/'
    label_df = pd.read_csv(folder_path + 'label.csv')

    # 将 label 数据转换为字典，以便快速查找
    label_dict = dict(zip(label_df['id'], label_df['label']))

    url_pattern = re.compile(r'https?://\S+')

    # 创建空列表来存储 bot 和 human 的 tweet 信息
    bot_tweets = []
    human_tweets = []

    with open(folder_path + 'tweet.json', 'r') as f:
        tweets = json.load(f)

    # 读取 tweet.json 文件
    for tweet in tweets:
        # 检查 tweet 的 author_id 是否在 label 字典中
        author_id = 'u' + str(tweet.get("author_id"))
        if author_id in label_dict:
            clean_text = re.sub(url_pattern, '', tweet.get("text", ""))
            # 根据 label 的类别存储 tweet 信息
            if tweet.get("entities") is not None:
                hashtags = tweet.get("entities").get("hashtags")
            else:
                hashtags = []
            tweet_data = {
                "author_id": tweet.get("author_id"),
                "id": tweet.get("id"),
                "created_at": tweet.get("created_at"),
                "text": clean_text.strip(),
                "hashtags": hashtags
            }
            
            if label_dict[author_id] == "bot":
                bot_tweets.append(tweet_data)
            elif label_dict[author_id] == "human":
                human_tweets.append(tweet_data)

    # 将 bot 和 human 的 tweet 信息保存为独立的文件
    with open(folder_path + 'bot_tweet.json', 'w') as f:
        json.dump(bot_tweets, f, indent=4)

    with open(folder_path + 'human_tweet.json', 'w') as f:
        json.dump(human_tweets, f, indent=4)

    print(f"{i} 分割完成：已生成 bot_tweet.json 和 human_tweet.json 文件。")
