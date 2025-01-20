import json
import pandas as pd
import matplotlib.pyplot as plt
from transformers import pipeline
from statistics import mean, stdev
from tqdm import tqdm

file_name = "ukraine_evobot"
# 1. 加载数据
with open(f"/home/fanqi/llm_simulation/HiSim/output/{file_name}.json", "r") as file:
    user_data = json.load(file)

# 2. 情感分析
classifier = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment",
    tokenizer="cardiffnlp/twitter-roberta-base-sentiment",
    device="cuda:2",
)

def analyze_sentiment(text):
    max_length = 500
    text = text[:max_length]
    # text = [" ".join(tokens[i:i + max_length]) for i in range(0, len(tokens), max_length)]
    result = classifier(text, truncation=True)[0]
    label = result["label"]
    score = result["score"]
    if label == "LABEL_2":  # Positive sentiment
        return score
    elif label == "LABEL_0":  # Negative sentiment
        return -score
    else:  # Neutral sentiment
        return 0

# 3. 处理每个用户的数据
all_quarterly_scores = []  # 用于存储所有用户的季度平均分
num_quarters = 18 # 每个用户固定9个季度

for user, user_responses in tqdm(user_data.items()):
    # 提取用户的所有 parsed_response
    parsed_responses = [response["parsed_response"] for response in user_responses.values() if "parsed_response" in response]

    # 每三条作为一个季度分组
    quarterly_scores = []
    for i in range(0, len(parsed_responses), 1):
        quarter_responses = parsed_responses[i:i+1]
        if quarter_responses:  # 确保非空
            sentiment_scores = [analyze_sentiment(response) for response in quarter_responses]
            quarterly_scores.append(mean(sentiment_scores))  # 存储当前季度的均值

    # 填充缺失的季度为 NaN
    while len(quarterly_scores) < num_quarters:
        quarterly_scores.append(float('nan'))

    all_quarterly_scores.append(quarterly_scores)

# 4. 汇总所有用户的季度情感得分
all_quarterly_scores = pd.DataFrame(all_quarterly_scores).T  # 转置，行为季度，列为用户
mean_scores = all_quarterly_scores.mean(axis=1, skipna=True)  # 按季度计算平均分
std_scores = all_quarterly_scores.std(axis=1, skipna=True)  # 按季度计算标准差

# 5. 绘制图表
quarters = [f"D{i+1}" for i in range(num_quarters)]

plt.figure(figsize=(10, 6))
plt.plot(quarters, mean_scores, label="Mean Sentiment Score", marker="o", color="blue")
plt.fill_between(
    quarters,
    mean_scores - 0.1*std_scores,
    mean_scores + 0.1*std_scores,
    color="blue",
    alpha=0.2,
    label="Standard Deviation"
)
plt.xticks(rotation=45)
plt.title("Sentiment Distribution by Quarter")
plt.xlabel("Quarter")
plt.ylabel("Sentiment Score (Weighted)")
plt.legend()
plt.tight_layout()
plt.show()
plt.savefig(f"/home/fanqi/llm_simulation/HiSim/output/{file_name}.pdf")

print(f"Bot version: {file_name}")
print(f"Mean scores: {mean_scores}\n")
print(f"Mean of Mean: {mean(mean_scores)}")
print(f"Std scores: {std_scores}\n")
print(f"Mean of std: {mean(std_scores)}")
