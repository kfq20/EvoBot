import matplotlib.pyplot as plt
import numpy as np

# Data

BC_mean = []
BC_std = []

evobot_mean = [-0.436,0.19974,-0.0823,-0.2839,-0.0308,-0.2335,0.09956,0.09089,0.17635,-0.2697,-0.2565,-0.0340,-0.2083,-0.3338,0.07918,0.12519,-0.2775,0.13014,]
evobot_std = [0.331904,0.602683,0.621506,0.392442,0.493974,0.417915,0.598510,0.524678,0.499822,0.345592,0.316015,0.732117,0.303297,0.257090,0.713503,0.620944,0.356444,0.513410,]
real_mean = [-0.046410,-0.101333,-0.072297,-0.068134,-0.069393,-0.051659,-0.049184,-0.061411,-0.055253,-0.080172,-0.054257,-0.053197,-0.069024,-0.028068,-0.109015,-0.018858,-0.022148,-0.044256]
real_std = [0.5820,0.68779,0.68825,0.67568,0.71493,0.64221,0.69712,0.67054,0.66634,0.62406,0.65230,0.69812,0.69180,0.64315,0.72270,0.69430,0.62860,0.68253,]
llama_std = [0.391883,0.424619,0.414699,0.496712,0.397862,0.507886,0.456327,0.494889,0.414349,0.381722,0.431743,0.385698,0.427147,0.354563,0.221890,0.370006,0.365914,0.359241]
llama_mean = [-0.798324, 0.154392,-0.215210,-0.570638,-0.583455,-0.573688, 0.504724,-0.501814,-0.514756,-0.727017,-0.793776, 0.034741,-0.790646,-0.828310,-0.286822,-0.274498,-0.729764,-0.560208,]

x = np.arange(len(real_mean))

# Calculate mean of means and std of stds
real_mean_avg = np.mean(real_mean)
real_std_avg = np.mean(real_std)
llama_mean_avg = np.mean(llama_mean)
llama_std_avg = np.mean(llama_std)
evobot_mean_avg = np.mean(evobot_mean)
evobot_std_avg = np.mean(evobot_std)

# Calculate deviations from Real
llama_deviation_mean = np.mean(np.abs(np.array(llama_mean) - np.array(real_mean)))
evobot_deviation_mean = np.mean(np.abs(np.array(evobot_mean) - np.array(real_mean)))

llama_deviation_std = np.mean(np.abs(np.array(llama_std) - np.array(real_std)))
evobot_deviation_std = np.mean(np.abs(np.array(evobot_std) - np.array(real_std)))

# Print calculations
print(f"Real: mean of means = {real_mean_avg:.4f}", ", mean of stds =", real_std_avg)
print(f"LLaMA: mean of means =", llama_mean_avg, ", mean of stds =", llama_std_avg)
print(f"EvoBot: mean of means =", evobot_mean_avg, ", mean of stds =", evobot_std_avg)
print(f"Deviation from Real (mean):")
print(f"  LLaMA =", llama_deviation_mean)
print(f"  EvoBot =", evobot_deviation_mean)

print("Deviation from Real (std):")
print(f"  LLaMA =", llama_deviation_std)
print(f"  EvoBot =", evobot_deviation_std)


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

# Customization
plt.title('Mean and Standard Deviation of Attitude Scores', fontsize=14)
plt.xlabel('Point Index', fontsize=12)
plt.ylabel('Scores', fontsize=12)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.tight_layout()

# Show plot
plt.show()
plt.savefig("/home/fanqi/llm_simulation/HiSim/output/ukraine/all_ukraine.pdf")