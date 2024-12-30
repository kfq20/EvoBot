import json
import pandas as pd

for comm in [10, 11]:
    # 加载 user.json
    with open(f"/home/fanqi/llm_simulation/data/raw_data/community_{comm}/user_info.json", "r", encoding="utf-8") as f:
        user_data = json.load(f)

    # 加载 label.csv
    label_data = pd.read_csv(f"/home/fanqi/llm_simulation/data/raw_data/community_{comm}/label.csv")

    # 筛选 human 的 id
    human_ids = set(label_data[label_data["label"] == "human"]["id"])

    # 筛选出 human 的用户信息
    human_users = [user for user in user_data if user["id"] in human_ids]

    # 保存到 human_user.json
    with open(f"/home/fanqi/llm_simulation/data/raw_data/community_{comm}/human_user_info.json", "w", encoding="utf-8") as f:
        json.dump(human_users, f, ensure_ascii=False, indent=4)

    print(f"共抽取了 {len(human_users)} 个 human 用户的信息，已保存到 human_user.json 文件中。")
