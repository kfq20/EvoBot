import pandas as pd
import json
for comm in range(12):
# 文件路径
    label_file = f'/home/fanqi/llm_simulation/data/raw_data/community_{comm}/label.csv'
    edge_file = f'/home/fanqi/llm_simulation/data/raw_data/community_{comm}/edge.csv'
    tweet_file = f'/home/fanqi/llm_simulation/data/raw_data/community_{comm}/tweet.json'

    # 读取label.csv文件
    label_df = pd.read_csv(label_file)
    user_count = len(label_df[label_df['label'] == 'human'])  # 用户数量
    bot_count = len(label_df[label_df['label'] == 'bot'])  # bot数量

    # 读取edge.csv文件
    edge_df = pd.read_csv(edge_file)
    edge_count = len(edge_df)  # 边的数量

    # 读取tweet.json文件
    with open(tweet_file, 'r', encoding='utf-8') as f:
        tweet_data = json.load(f)
    tweet_count = len(tweet_data)  # tweet数量

    # 输出统计结果
    print(f"{comm}")
    print(f"human数量: {user_count}")
    print(f"Bot数量: {bot_count}")
    print(f"Edge数量: {edge_count}")
    print(f"Tweet数量: {tweet_count}")
    print("\n")