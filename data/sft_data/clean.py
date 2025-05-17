import json
import re

# 定义正则表达式，匹配 http 或 https 开头到下一个换行符的内容
pattern = re.compile(r'https?.*?(?=\n|$)', re.DOTALL)
COMM = 11
# 打开并逐行处理 JSON 文件
processed_data = []
with open(f'./data/sft_data/community_{COMM}/sft_data.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():  # 确保行不为空
            try:
                # 解析每行 JSON
                data = json.loads(line)
                # 遍历字典的键值对
                for key, value in data.items():
                    if isinstance(value, str):  # 如果值是字符串，才进行处理
                        data[key] = re.sub(pattern, '', value)
                processed_data.append(data)
            except json.JSONDecodeError as e:
                print(f"解析 JSON 出错: {e}，出错行: {line}")

# 将处理后的数据写入新文件（逐行保存）
with open(f'./data/sft_data/community_{COMM}/sft_data.jsonl', 'w', encoding='utf-8') as f:
    for entry in processed_data:
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')

print("处理完成，结果已保存到 processed_file.json 文件中！")