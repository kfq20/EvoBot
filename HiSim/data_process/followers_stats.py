import pandas as pd
import json
# 读取 edge.csv 文件
edge_file_path = '/home/fanqi/llm_simulation/data/raw_data/community_5/edge.csv'  # 请将其替换为实际文件路径
user_file_path = '/home/fanqi/llm_simulation/data/raw_data/community_5/user.json'  # 请将其替换为实际文件路径

# 加载 edge 数据
edge_data = pd.read_csv(edge_file_path)

# 加载 user.json 数据
with open(user_file_path, 'r') as f:
    user_list = set(json.load(f))  # 将 user.json 转换为集合，方便筛选

# 筛选 relation 为 followers 的记录，且 source_id 和 target_id 都在 user_list 中
filtered_data = edge_data[
    (edge_data['relation'] == 'followers') &
    (edge_data['source_id'].isin(user_list)) &
    (edge_data['target_id'].isin(user_list))
]

# 构建字典，统计每个 source_id 对应的 followers
followers_dict = (
    filtered_data.groupby('source_id')['target_id']
    .apply(list)
    .to_dict()
)

# 将结果保存为 JSON 文件
output_file = '/home/fanqi/llm_simulation/data/raw_data/community_5/filtered_followers.json'
with open(output_file, 'w') as f:
    json.dump(followers_dict, f, ensure_ascii=False, indent=4)

print(f"JSON 文件已成功保存到 {output_file}")
