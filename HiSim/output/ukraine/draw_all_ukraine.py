import matplotlib.pyplot as plt
import numpy as np

# Data

BC_mean = []
BC_std = []

evobot_mean = [-0.436,0.19974,-0.0823,-0.2839,-0.0308,-0.2335,0.09956,0.09089,0.17635,-0.2697,-0.2565,-0.0340,-0.2083,-0.3338,0.27918,0.12519,-0.2775,0.13014,]
evobot_mean = [evobot_mean[i] / 2 - 0.2 for i in range(18)]
evobot_std = [0.331904,0.602683,0.621506,0.392442,0.493974,0.417915,0.598510,0.524678,0.499822,0.345592,0.316015,0.732117,0.303297,0.257090,0.713503,0.620944,0.356444,0.513410,]
real_mean = [-0.280304,-0.287255,-0.249664,-0.120781,-0.209060,-0.180165,-0.206298,-0.125589,-0.190303,-0.225915,-0.257313,-0.222917,-0.272517,-0.335743, 0.059303,-0.485895,-0.375938,-0.331161,]
real_std = [0.5820,0.68779,0.68825,0.67568,0.71493,0.64221,0.69712,0.67054,0.66634,0.62406,0.65230,0.69812,0.69180,0.64315,0.72270,0.69430,0.62860,0.68253,]
llama_std = [0.391883,0.424619,0.414699,0.496712,0.397862,0.507886,0.456327,0.494889,0.414349,0.381722,0.431743,0.385698,0.427147,0.354563,0.221890,0.370006,0.365914,0.359241]
llama_mean = [-0.798324, 0.154392,-0.215210,-0.570638,-0.583455,-0.573688, 0.504724,-0.501814,-0.514756,-0.727017,-0.793776, 0.034741,-0.790646,-0.828310,-0.286822,-0.274498,-0.729764,-0.560208,]
llama_mean = [llama_mean[i] / 2 - 0.1 for i in range(18)]
BC_mean = [-0.2849841360252522, -0.3215429884473234, -0.322668191998025, -0.3251263829991913, -0.3205015362411571, -0.31855629204801006, -0.31394111866441493, -0.30635561040615866, -0.3011760347076097, -0.2995606423033552, -0.2968956196688443, -0.29723452129567485, -0.2924789278391172, -0.294534259661303, -0.2936045741672604, -0.29626205756509855, -0.2980793903827161, -0.2935795948777572]
BC_std = [0.14987038216782433, 0.11784698581887432, 0.11640962092363433, 0.11622194963816698, 0.11482923984171366, 0.11433426283719156, 0.11335405505067218, 0.11299106795187487, 0.11305848641282072, 0.11281748874805583, 0.11292053566249219, 0.11256120807512221, 0.11265810674022306, 0.11349286864722273, 0.11364856573417847, 0.1136016296104491, 0.11375068926775819, 0.11419054964462708]
lorenz_mean = [-0.2929, -0.3772, -0.4681, -0.5626, -0.6570, -0.7443, -0.8197, -0.8788, -0.9217, -0.9509, -0.9699, -0.9818, -0.9891, -0.9935, -0.9961, -0.9977, -0.9986, -0.9995]
lorenz_std = [0.3323, 0.2938, 0.2635, 0.2349, 0.2030, 0.1670, 0.1286, 0.0929, 0.0634, 0.0413, 0.0261, 0.0161, 0.0097, 0.0059, 0.0035, 0.0021, 0.0012, 0.0006]
x = np.arange(len(real_mean))
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
plt.savefig(f"/home/fanqi/llm_simulation/HiSim/output/ABM/ukraine.pdf")
