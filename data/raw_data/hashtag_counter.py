# import json
# from collections import Counter

# # 读取 JSON 文件
# input_file = "/home/fanqi/llm_simulation/TwiBot-22/src/BotRGCN/datasets/Twibot-22/tweet_0-3.json"
# output_file = "/home/fanqi/llm_simulation/TwiBot-22/src/BotRGCN/datasets/Twibot-22/hashtags_frequency.json"

# # 用于存储所有文本的 Counter
# hashtags_counter = Counter()

# # 读取 JSON 数据并提取信息
# with open(input_file, 'r', encoding='utf-8') as f:
#     data = json.load(f)  # 读取并解析整个 JSON 文件为 Python 对象
    
#     # 遍历每条数据
#     for record in data:
#         # 每行是一个嵌套的字典
#         entity = record.get("entities", {})
#         if entity is not None:
#             hashtags = entity.get("hashtags", [])
        
#         # 提取每个非空的 "text" 字段
#         for hashtag in hashtags:
#             if isinstance(hashtag, dict) and "text" in hashtag:
#                 hashtags_counter[hashtag["text"].lower()] += 1  # 转换为小写，避免大小写重复

# # 按频率排序
# sorted_hashtags = hashtags_counter.most_common()

# # 保存排序后的结果到文件
# with open(output_file, 'w', encoding='utf-8') as f:
#     json.dump(sorted_hashtags, f, ensure_ascii=False, indent=4)

# print(f"Hashtags frequency has been saved to {output_file}")


import json
# from langdetect import detect, LangDetectException
# from googletrans import Translator
from collections import Counter

# 创建翻译器实例
# translator = Translator()

for comm in [5]:
# 读取 JSON 文件
    input_file = f"/home/fanqi/llm_simulation/data/raw_data/community_{comm}/tweet.json"
    output_file = f"/home/fanqi/llm_simulation/data/raw_data/community_{comm}/translated_hashtags.json"

    # 用于存储所有文本的 Counter
    hashtags_counter = Counter()

    # 读取完整 JSON 数据
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)  # 读取并解析整个 JSON 文件为 Python 对象
        
        # 遍历每条数据
        for record in data:
            # 访问 "entities" -> "hashtags" 字段
            entity = record.get("entities", {})
            if entity is not None:
                hashtags = entity.get("hashtags", [])
            
            # 提取每个非空的 "text" 字段
            for hashtag in hashtags:
                if isinstance(hashtag, dict) and "text" in hashtag:
                    original_text = hashtag["text"]
                    
                    # 检测语言
                    # try:
                    #     detected_lang = detect(original_text)
                    # except LangDetectException:
                    #     detected_lang = 'unknown'
                    
                    # # 如果不是英语（en），则翻译成英语
                    # if detected_lang != 'en':
                    #     translated = translator.translate(original_text, src=detected_lang, dest='en').text
                    #     hashtag["text"] = translated  # 更新为翻译后的文本

                    # 统计翻译后的或原始的文本
                    hashtags_counter[hashtag["text"].lower()] += 1  # 转换为小写，避免大小写重复

    # 按频率排序
    sorted_hashtags = hashtags_counter.most_common()

    # 保存排序后的结果到文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(sorted_hashtags, f, ensure_ascii=False, indent=4)

    print(f"Translated hashtags frequency has been saved to {output_file}")
