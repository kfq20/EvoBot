import pandas as pd
import json
# 读取 edge.csv 文件
COMM = 5
edge_file_path = f'/home/fanqi/llm_simulation/data/raw_data/community_{COMM}/edge.csv'  # 请将其替换为实际文件路径
user_file_path = f'/home/fanqi/llm_simulation/data/raw_data/community_{COMM}/user.json'  # 请将其替换为实际文件路径

# 加载 edge 数据
edge_data = pd.read_csv(edge_file_path)

# 加载 user.json 数据
with open(user_file_path, 'r') as f:
    user_list = set(json.load(f))  # 将 user.json 转换为集合，方便筛选

# 筛选 relation 为 followers 的记录，且 source_id 和 target_id 都在 user_list 中
filtered_data = edge_data[
    (edge_data['relation'] == 'following') & 
    (edge_data['source_id'].isin(user_list)) &
    (edge_data['target_id'].isin(user_list))
]

# 构建字典，统计每个 source_id 对应的 followers
followers_dict = (
    filtered_data.groupby('source_id')['target_id']
    .apply(list)
    .to_dict()
)
followers_dict_2  = (
    filtered_data.groupby('target_id')['source_id']
    .apply(list)
    .to_dict()
)

# 筛选 relation 为 followers 的记录，且 source_id 和 target_id 都在 user_list 中
following_data = edge_data[
    (edge_data['relation'] == 'followers') & 
    (edge_data['source_id'].isin(user_list)) &
    (edge_data['target_id'].isin(user_list))
]

# 构建字典，统计每个 source_id 对应的 followers
following_dict = (
    following_data.groupby('target_id')['source_id']
    .apply(list)
    .to_dict()
)

following_dict_2 = (
    following_data.groupby('source_id')['target_id']
    .apply(list)
    .to_dict()
)

# 遍历 followings_dict，将其补充到 followers_dict 中
for key, value in following_dict.items():
    if key in followers_dict:
        followers_dict[key].extend(value)  # 如果 key 存在，将 value 合并
    else:
        followers_dict[key] = value  # 如果 key 不存在，直接添加

for key, value in following_dict_2.items():
    if key in followers_dict:
        followers_dict[key].extend(value)  # 如果 key 存在，将 value 合并
    else:
        followers_dict[key] = value  # 如果 key 不存在，直接添加

for key, value in followers_dict_2.items():
    if key in followers_dict:
        followers_dict[key].extend(value)  # 如果 key 存在，将 value 合并
    else:
        followers_dict[key] = value  # 如果 key 不存在，直接添加
# 打印合并后的字典
print(followers_dict)

# 将结果保存为 JSON 文件
output_file = f'/home/fanqi/llm_simulation/HiSim/example_data/following_comm{COMM}.json'
with open(output_file, 'w') as f:
    json.dump(followers_dict, f, ensure_ascii=False, indent=4)

print(f"JSON 文件已成功保存到 {output_file}")

# # 读取 user.json，将 id 映射为 name
# with open('/home/fanqi/llm_simulation/data/raw_data/community_7/user_info.json', 'r', encoding='utf-8') as user_file:
#     user_data = json.load(user_file)

# id_to_name = {user['id']: user['name'] for user in user_data}

# # 读取 follower.json，将 id 替换为 name
# with open('/home/fanqi/llm_simulation/HiSim/example_data/superbowl_followers.json', 'r', encoding='utf-8') as follower_file:
#     follower_data = json.load(follower_file)

# # 创建一个新的字典，将 id 转换为对应的 name
# follower_with_names = {}
# for key, follower_ids in follower_data.items():
#     # 使用 key 对应的 name 作为新的 key
#     new_key = id_to_name.get(key, key)  # 如果找不到对应的 name，则使用原 id 作为 key
#     # 将 follower_ids 中的 id 转换为对应的 name
#     new_follower_ids = [id_to_name.get(follower_id, follower_id) for follower_id in follower_ids]
#     follower_with_names[new_key] = new_follower_ids

# # 输出新的 follower 数据
# with open('/home/fanqi/llm_simulation/HiSim/example_data/superbowl_followers.json', 'w', encoding='utf-8') as output_file:
#     json.dump(follower_with_names, output_file, ensure_ascii=False, indent=4)

# print("Conversion complete!")