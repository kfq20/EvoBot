import json

COMM = 7
tweet_id = 8
# 加载 JSON 文件
def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

# 保存为 JSON 文件
def save_json(data, output_path):
    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

# 合并 JSON 列表
def merge_json_lists(file1, file2, output_file):
    # 加载两个 JSON 文件
    list1 = load_json(file1)
    list2 = load_json(file2)
    
    # 合并列表
    merged_list = list1 + list2
    
    # 保存合并结果
    save_json(merged_list, output_file)

for tweet_id in [5,6,7,8]:
    # 文件路径
    file1_path = f'/home/fanqi/llm_simulation/data/raw_data/community_{COMM}/tweet.json'
    file2_path = f'/home/fanqi/llm_simulation/data/raw_data/community_{COMM}/tweet_{tweet_id}.json'
    output_path = f'/home/fanqi/llm_simulation/data/raw_data/community_{COMM}/tweet.json'

    merge_json_lists(file1_path, file2_path, output_path)
print(f"合并完成，结果已保存到 {output_path}")