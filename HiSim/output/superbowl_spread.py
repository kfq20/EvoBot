import json
from datetime import datetime, timedelta
from collections import defaultdict
import matplotlib.pyplot as plt

with open("/home/fanqi/llm_simulation/HiSim/output/superbowl_evobot.json", "r", encoding="utf-8") as f:
    results = json.load(f)
output_folder = "/home/fanqi/llm_simulation/HiSim/output/"
keywords = {"champion", "ram"}
keywords2 = {"ram", "win"}

# 初始化变量
unique_authors = set()  # 存储累计提到关键词的作者
cumulative_counts = []  # 每个时间点的累计作者数量

# 遍历时间点
for time_point in range(21):  # 0 到 20
    for author, time_dict in results.items():
        # 检查作者在当前时间点的 parsed_response
        if str(time_point) in time_dict:
            response = time_dict[str(time_point)].get("parsed_response", "")
            # 判断关键词是否出现在 response 中（忽略大小写）
            if all(keyword in response.lower() for keyword in keywords) or all(keyword in response.lower() for keyword in keywords2):
                unique_authors.add(author)  # 累计作者
    
     # 记录当前时间点的累计作者数量
    cumulative_counts.append(len(unique_authors))
cumulative_counts = [num // 5 for num in cumulative_counts]
print(cumulative_counts)
plt.figure(figsize=(10, 6))
plt.plot(range(21), cumulative_counts, marker='o', linestyle='-', color='b', label="Cumulative Authors")
plt.xticks(range(21))
plt.xlabel("Time Window")
plt.ylabel("Cumulative User Count")
plt.title("umulative User Count Discussing Super Bowl LVI")
plt.legend()

plt.savefig(output_folder + "superbowl_cumulative_author_evobot.png", dpi=500)  # 保存为文件