# import json

# # 加载 JSON 文件
# with open('/home/fanqi/llm_simulation/data/processed_data/community_2/id_tweet.json', 'r', encoding='utf-8') as file:
#     data = json.load(file)

# # 初始化计数器
# total_length = 0
# string_count = 0

# # 遍历 JSON 的值
# for value in data.values():
#     if isinstance(value, list):
#         for string in value:
#             if isinstance(string, str):
#                 total_length += len(string)
#                 string_count += 1

# # 计算平均长度
# if string_count > 0:
#     average_length = total_length / string_count
#     print(f"平均字符串长度为: {average_length}")
# else:
#     print("未找到字符串数据。")

import json

# 加载 JSON 文件
with open('/home/fanqi/llm_simulation/data/processed_data/community_5/id_tweet_dpo_2_long.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# 截断字符串的函数
def truncate_string(s, max_length=512):
    return s[:max_length] if len(s) > max_length else s

# 遍历 JSON 数据并截断长字符串
for key, value in data.items():
    if isinstance(value, list):
        data[key] = [
            truncate_string(string) if isinstance(string, str) else string
            for string in value
        ]

# 保存处理后的 JSON 文件
with open('/home/fanqi/llm_simulation/data/processed_data/community_5/id_tweet_dpo_2_short.json', 'w', encoding='utf-8') as file:
    json.dump(data, file, ensure_ascii=False, indent=4)

print("处理完成，所有字符串超过512字符的部分已截断并保存到 tweet_truncated.json。")

