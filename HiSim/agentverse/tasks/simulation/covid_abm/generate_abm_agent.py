import json
import random
import yaml
import numpy as np

# 加载user.json文件
with open('/home/fanqi/llm_simulation/data/raw_data/community_2/user.json', 'r', encoding='utf-8') as f:
    users = json.load(f)

'''COVID para'''
# mean = -0.065231
# std_dev = 0.435328

'''Ukraine para'''
mean = -0.046410
std_dev = 0.39188

# 设置随机种子以确保可复现
np.random.seed(42)

# 生成初始化的init_att，并确保在[-1, 1]范围内
def generate_init_att():
    while True:
        sample = np.random.normal(mean, std_dev)
        if -1 <= sample <= 1:
            return sample

# 创建YAML数据结构
abm_model = {
    'model_type': 'bcm',
    'order': 'random',
    'alpha': 0.3,
    'bc_bound': 0.1,
    'agent_config_lst': []
}

# 填充agent_config_lst
for i, user in enumerate(users[:len(users)]):  # 假设你想用user.json中的所有用户
    agent_config = {
        'id': i,
        'init_att': generate_init_att(),
        'name': user
    }
    abm_model['agent_config_lst'].append(agent_config)

# 输出到YAML文件
with open('/home/fanqi/llm_simulation/HiSim/agentverse/tasks/simulation/covid_abm/abm_model_comm2.yaml', 'w', encoding='utf-8') as f:
    yaml.dump(abm_model, f, allow_unicode=True, default_flow_style=False)

print("YAML文件生成完成！")