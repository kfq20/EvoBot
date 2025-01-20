import yaml
import numpy as np

# 假设你的 YAML 文件路径是 'data.yaml'
with open('/home/fanqi/llm_simulation/HiSim/output/ABM/covid_abm.yaml', 'r') as file:
    data = yaml.safe_load(file)

# 统计每个 opinion_results_1 到 opinion_results_27 的 mean 和 std
for i in range(1, 28):  # From opinion_results_1 to opinion_results_27
    key = f'opinion_results_{i}'
    
    if key in data:  # Ensure the key exists in the data
        values = list(data[key].values())
        
        # Calculate mean and std
        mean_value = np.mean(values)
        std_value = np.std(values)
        
        # Print results
        print(f"{mean_value:.4f} {std_value:.4f}")
    else:
        print(f"{key} not found in the data.")
