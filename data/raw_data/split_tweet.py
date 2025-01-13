import json

tweet_index = 7
COMM = 7
# 文件路径
for tweet_index in [8,7,6,5]:
    tweet_file = f"/home/fanqi/llm_simulation/data/raw_data/tweet_{tweet_index}.json"
    user_file = f"/home/fanqi/llm_simulation/data/raw_data/community_{COMM}/user.json"
    output_file = f"/home/fanqi/llm_simulation/data/raw_data/community_{COMM}/tweet_{tweet_index}.json"

    # 加载 user.json
    with open(user_file, "r", encoding="utf-8") as f:
        user_ids = set(json.load(f))  # 加载为集合以便快速查询

    # 加载并筛选 tweet.json
    filtered_tweets = []
    with open(tweet_file, "r", encoding="utf-8") as f:
        tweets = json.load(f)  # 假设 tweet.json 是一个 JSON 数组
        for tweet in tweets:
            if 'u' + str(tweet.get("author_id")) in user_ids:
                filtered_tweets.append(tweet)

    # 保存结果到新的 JSON 文件
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(filtered_tweets, f, ensure_ascii=False, indent=4)

print(f"已成功筛选推文，结果保存到 {output_file}")