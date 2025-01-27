import yaml
import numpy as np

# 加载 YAML 文件
with open('/home/fanqi/llm_simulation/HiSim/output/ABM/ukraine_lorenz.yaml', 'r') as file:
    data = yaml.safe_load(file)

# 提取前 100 个 opinion_results 数据
results = []
for i in range(1, 101):  # 从 opinion_results_1 到 opinion_results_100
    key = f'opinion_results_{i}'
    if key in data:
        values = list(data[key].values())
        results.append(values)
    else:
        print(f"{key} not found in the data.")

# 将数据转换为 numpy 数组
results = np.array(results)

# 将 100 组数据缩减为 18 组
num_groups = 18
group_size = results.shape[0] // num_groups  # 每组的大小
reduced_results = []

for i in range(num_groups):
    start = i * group_size
    end = (i + 1) * group_size if i < num_groups - 1 else results.shape[0]  # 最后一组包含剩余数据
    group_data = results[start:end]
    
    # 计算每组的 mean 和 std
    group_mean = np.mean(group_data)
    group_std = np.std(group_data)
    reduced_results.append((group_mean, group_std))

# 输出最终的 mean 和 std
final_means = [result[0] for result in reduced_results]
final_stds = [result[1] for result in reduced_results]

print("Final means: ", [f"{x:.4f}" for x in final_means])
print("Final stds: ", [f"{x:.4f}" for x in final_stds])