import pickle

# 指定 pkl 文件的路径
file_path = "/home/fanqi/llm_simulation/HiSim/ckptenvironment.pkl"

# 打开并加载 pkl 文件
with open(file_path, "rb") as file:  # 使用 "rb" 模式读取二进制文件
    data = pickle.load(file)

# 打印或处理数据
print(data)