import json
import numpy as np

# 假设你的 JSON 文件路径是 'data.json'
with open('/home/fanqi/llm_simulation/HiSim/output/ABM/covid_abm.json', 'r') as file:
    data = json.load(file)

# 获取所有的值
values = list(data['opinion_results'].values())

# 计算均值和标准差
mean_value = np.mean(values)
std_value = np.std(values)

print(f"Mean: {mean_value}")
print(f"Standard Deviation: {std_value}")
