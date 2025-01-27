import matplotlib.pyplot as plt
import numpy as np

# Data
x = np.arange(9)
real_mean = [-0.065231, 0.013033, 0.015412, -0.063455, 0.057049, 0.143577, -0.132960, -0.039737, -0.078328]
real_std = [0.435328, 0.406997, 0.552411, 0.462709, 0.478888, 0.540539, 0.457606, 0.451614, 0.464655]
llama_mean = [0.012197, -0.050299, -0.149965, -0.078737, 0.006963, 0.007470, -0.044005, 0.000710, -0.321432]
llama_std = [0.399814, 0.346465, 0.371428, 0.400848, 0.317927, 0.403074, 0.315060, 0.391852, 0.362248]
evobot_mean = [0.037269, -0.077501, -0.021144, 0.002854, 0.101113, 0.059695, -0.182722, 0.079995, -0.135330]
evobot_std = [0.442057, 0.415695, 0.423705, 0.435475, 0.398883, 0.446137, 0.407517, 0.432203, 0.412800]
BC_mean = [-0.026,-0.003,0.006066667,0.011633333,0.013466667,0.014633333,0.015166667,0.015166667, 0.0154]
BC_std = [0.313533333,0.2356,0.187633333,0.156533333,0.1297,0.107766667,0.088433333,0.069333333,0.054933333]
lorenz_mean = [0.003722218584115507, 0.05344046912568292, 0.09358489098786514, 0.12142988172442203, 0.1297039293965979, 0.11503960074583712, 0.0938758404034093, 0.07631570664085117, 0.06522797671896527]
lorenz_std = [0.39115387597522505, 0.4447508700752849, 0.5481151609132431, 0.6616940156837786, 0.769130655655448, 0.8587048385038312, 0.9207902336657349, 0.9570759526992437, 0.9753602832244381]
# Calculate mean of means and std of stds
real_mean_avg = np.mean(real_mean)
real_std_avg = np.mean(real_std)
llama_mean_avg = np.mean(llama_mean)
llama_std_avg = np.mean(llama_std)
evobot_mean_avg = np.mean(evobot_mean)
evobot_std_avg = np.mean(evobot_std)
BC_mean_avg = np.mean(BC_mean)
BC_std_avg = np.mean(BC_std)
lorenz_mean_avg = np.mean(lorenz_mean)
lorenz_std_avg = np.mean(lorenz_std)
# Calculate deviations from Real
llama_deviation_mean = np.mean(np.abs(np.array(llama_mean) - np.array(real_mean)))
evobot_deviation_mean = np.mean(np.abs(np.array(evobot_mean) - np.array(real_mean)))
BC_deviation_mean = np.mean(np.abs(np.array(BC_mean) - np.array(real_mean)))
lorenz_deviation_mean = np.mean(np.abs(np.array(lorenz_mean) - np.array(real_mean)))

llama_deviation_std = np.mean(np.abs(np.array(llama_std) - np.array(real_std)))
evobot_deviation_std = np.mean(np.abs(np.array(evobot_std) - np.array(real_std)))
BC_deviation_std = np.mean(np.abs(np.array(BC_std) - np.array(real_std)))
lorenz_deviation_std = np.mean(np.abs(np.array(lorenz_std) - np.array(real_std)))

# Print calculations
print(f"Real: mean of means = {real_mean_avg:.4f}", ", mean of stds =", real_std_avg)
print(f"LLaMA: mean of means =", llama_mean_avg, ", mean of stds =", llama_std_avg)
print(f"EvoBot: mean of means =", evobot_mean_avg, ", mean of stds =", evobot_std_avg)
print(f"BC: mean of means =", BC_mean_avg, ", mean of stds =", BC_std_avg)
print(f"Lorenz: mean of means =", lorenz_mean_avg, ", mean of stds =", lorenz_std_avg)
print(f"Deviation from Real (mean):")
print(f"  LLaMA =", llama_deviation_mean)
print(f"  EvoBot =", evobot_deviation_mean)
print(f"  BC =", BC_deviation_mean)
print(f"  Lorenz =", lorenz_deviation_mean)

print("Deviation from Real (std):")
print(f"  LLaMA =", llama_deviation_std)
print(f"  EvoBot =", evobot_deviation_std)
print(f"  BC =", BC_deviation_std)
print(f"  Lorenz =", lorenz_deviation_std)

# Plot
plt.figure(figsize=(10, 6))

# Real data
plt.plot(x, real_mean, label='Real', color='blue', linestyle='-')
plt.fill_between(x, np.array(real_mean) - 0.1*np.array(real_std), np.array(real_mean) + 0.1*np.array(real_std), color='blue', alpha=0.2)

# Llama data
plt.plot(x, llama_mean, label='LLaMA', color='green', linestyle='-')
plt.fill_between(x, np.array(llama_mean) - 0.1*np.array(llama_std), np.array(llama_mean) + 0.1*np.array(llama_std), color='green', alpha=0.2)

# EvoBot data
plt.plot(x, evobot_mean, label='EvoBot', color='red', linestyle='-')
plt.fill_between(x, np.array(evobot_mean) - 0.1*np.array(evobot_std), np.array(evobot_mean) + 0.1*np.array(evobot_std), color='red', alpha=0.2)

# BC data
plt.plot(x, BC_mean, label='BC', color='orange', linestyle='-')
plt.fill_between(x, np.array(BC_mean) - 0.1*np.array(BC_std), np.array(BC_mean) + 0.1*np.array(BC_std), color='orange', alpha=0.2)

# Lorenz data
plt.plot(x, lorenz_mean, label='Lorenz', color='purple', linestyle='-')
plt.fill_between(x, np.array(lorenz_mean) - 0.1*np.array(lorenz_std), np.array(lorenz_mean) + 0.1*np.array(lorenz_std), color='purple', alpha=0.2)

# Customization
plt.title('Mean and Standard Deviation of Attitude Scores', fontsize=14)
plt.xlabel('Point Index', fontsize=12)
plt.ylabel('Scores', fontsize=12)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.tight_layout()

# Show plot
plt.show()
plt.savefig("/home/fanqi/llm_simulation/HiSim/output/all_covid.pdf")