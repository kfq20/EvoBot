import pickle
import json

# 指定文件路径
task = "covid_abm"
pkl_file_path = f"/home/fanqi/llm_simulation/HiSim/output/{task}.pkl"
json_file_path = f"/home/fanqi/llm_simulation/HiSim/output/{task}.json"

# 读取 pickle 文件
with open(pkl_file_path, "rb") as pkl_file:
    try:
        data = pickle.load(pkl_file)  # 读取 pickle 数据
    except ModuleNotFoundError as e:
        print(f"模块缺失：{e}")
        data = None

# 检查数据是否成功加载
if data is not None:
    # 将数据转换为 JSON 可序列化的格式（如果需要）
    def convert_to_serializable(obj):
        """递归将复杂对象转换为可序列化的格式"""
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        else:
            # 如果遇到不支持的类型，可以选择转换为字符串或忽略
            return str(obj)

    json_data = convert_to_serializable(data)

    # 保存为 JSON 文件
    with open(json_file_path, "w", encoding="utf-8") as json_file:
        json.dump(json_data, json_file, ensure_ascii=False, indent=4)
    print(f"数据已保存为 {json_file_path}")
else:
    print("无法加载 pickle 文件，检查模块依赖或文件内容。")
