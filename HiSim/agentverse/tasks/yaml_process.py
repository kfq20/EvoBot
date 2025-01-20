import yaml
import json

COMM = 7

def expand_yaml_with_json(yaml_file, summary_file, info_file, output_file):
    # 读取 YAML 样例
    with open(yaml_file, 'r') as yf:
        yaml_data = yaml.safe_load(yf)
    
    # 确保 agents 存在
    if 'agents' not in yaml_data or not isinstance(yaml_data['agents'], list):
        raise ValueError("YAML 文件中缺少 'agents' 列表")
    
    # 获取样例 agent 的模板
    sample_agent = yaml_data['agents'][0]
    
    # 读取 JSON 文件
    with open(info_file, 'r') as jf:
        human_info = json.load(jf)
    with open(summary_file, 'r') as sf:
        descriptions_data = json.load(sf)

    id_to_description = {entry['id']: entry['description'] for entry in descriptions_data}
    
    # 创建新的 agents 列表
    new_agents = []
    for agent in human_info:
        new_agent = sample_agent.copy()  # 使用样例 agent 的模板
        name = agent['name']
        new_agent['name'] = name  # 替换 name
        agent_id = agent['id']
        role_description = id_to_description[agent_id]
        role_description = f"You are a Twitter user named {name}. Here is a brief description about you: " + role_description
        new_agent['role_description'] = role_description # 替换 role_description
        new_agents.append(new_agent)
    
    # 更新 YAML 数据
    yaml_data['agents'] = new_agents
    
    # 写入新的 YAML 文件
    with open(output_file, 'w') as of:
        yaml.dump(yaml_data, of, allow_unicode=True)
    print(f"新的 YAML 文件已生成: {output_file}")

# 示例用法

# expand_yaml_with_json(yaml_file="/home/fanqi/llm_simulation/HiSim/agentverse/tasks/simulation/roe_macro_llm/config.yaml", 
#                       summary_file=f"/home/fanqi/llm_simulation/data/raw_data/community_{COMM}/user_summary.json", 
#                       info_file=f"/home/fanqi/llm_simulation/data/raw_data/community_{COMM}/user_info.json",
#                       output_file=f"/home/fanqi/llm_simulation/HiSim/agentverse/tasks/simulation/roe_macro_llm/agent_{COMM}.yaml")
# exit()
# import yaml
# import json
# with open(f"/home/fanqi/llm_simulation/HiSim/agentverse/tasks/simulation/roe_macro_llm/agent_{COMM}.yaml", 'r') as yf:
#     yaml_data = yaml.safe_load(yf)
# new_prompt_template = "Now you are acting as an agent named ${agent_name} in the social media Twitter. Here are some information:\n\n (1) The agent's description: ${role_description}\n\n  \
#     \  (2) Current time is ${current_time}\n\n    (3) The news you got is \"${trigger_news}\"\
#     \n\n    (4) The events that occurred in the past are ${chat_history}\n\n    (5) The twitter page you\
#     \ can see is ${tweet_page}\n\n    Use the information to assess if the user is interested in this news. If the user is, compose a tweet expressing an opinion. If not, write a random tweet."
# for agent in yaml_data["agents"]:
#     agent['prompt_template'] = new_prompt_template

# with open(f"/home/fanqi/llm_simulation/HiSim/agentverse/tasks/simulation/roe_macro_llm/config_{COMM}.yaml", 'w') as of:
#     yaml.dump(yaml_data, of, allow_unicode=True)

import yaml

def remove_yaml_anchors(input_file, output_file):
    # 自定义的 YAML 转换器，用于去除锚点和引用
    class NoAliasDumper(yaml.Dumper):
        def ignore_aliases(self, data):
            return True

    # 读取 YAML 文件
    with open(input_file, 'r') as infile:
        data = yaml.safe_load(infile)

    # 将数据重新写入 YAML 文件，禁用锚点和引用
    with open(output_file, 'w') as outfile:
        yaml.dump(data, outfile, Dumper=NoAliasDumper, allow_unicode=True)

    print(f"已生成无锚点的 YAML 文件: {output_file}")

# 示例用法
remove_yaml_anchors(
    input_file=f"/home/fanqi/llm_simulation/HiSim/agentverse/tasks/simulation/roe_macro_llm/config_{COMM}.yaml",    # 输入的 YAML 文件路径
    output_file=f"/home/fanqi/llm_simulation/HiSim/agentverse/tasks/simulation/roe_macro_llm/config_{COMM}.yaml"   # 输出的 YAML 文件路径
)
