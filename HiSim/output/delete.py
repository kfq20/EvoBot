import json

# 1. 加载 JSON 文件
input_file = "/home/fanqi/llm_simulation/HiSim/output/result.json"  # 输入文件路径
output_file = "/home/fanqi/llm_simulation/HiSim/output/result.json"  # 输出文件路径

with open(input_file, "r") as file:
    data = json.load(file)  # 加载 JSON 数据

# 2. 确保数据是一个列表
if isinstance(data, list):
    # 删除前 1080 个元素
    modified_data = data[1080:]
elif isinstance(data, dict):
    # 如果是字典，将其转为列表操作，然后再转换回字典
    modified_data = dict(list(data.items())[1080:])
else:
    raise TypeError("JSON 数据既不是列表也不是字典，无法操作。")

# 3. 保存到新的 JSON 文件
with open(output_file, "w") as file:
    json.dump(modified_data, file, indent=4)

print(f"已将删除前 1080 个元素后的数据保存到 {output_file}")