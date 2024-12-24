import pandas as pd
import json

# 读取selected_data.csv
selected_data = pd.read_csv('/home/fanqi/llm_simulation/data/user_edge.csv')

# 提取csv中的id
selected_ids = selected_data['id']

# 读取json文件
with open('./data/sub-communities/user0.json', 'r', encoding='utf-8') as f:
    users = json.load(f)

# 过滤数据
filtered_users = [user for user in users if str(user['owner_id']) in selected_ids.values]

# 保存结果到新的json文件
with open('./data/list_small.json', 'w', encoding='utf-8') as f:
    json.dump(filtered_users, f, ensure_ascii=False, indent=4)