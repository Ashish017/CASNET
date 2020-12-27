from parameters import settings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("Reacher_expert_PPO_data.csv")

training_envs = ["Reacher-v10","Reacher-v20","Reacher-v30","Reacher-v40","Reacher-v50"]
smoothing_size = 8

colors = ["blue","green","red","orange", "purple"]

for env_num in range(len(training_envs)):
	env = training_envs[env_num]
	for seed in range(1, settings.num_seeds+1):
		
		if seed == 1:
			env_mean = np.array(data[env +"_seed_" + str(seed)])
			env_max = env_mean
			env_min = env_mean
		else:
			env_mean = np.array(data[env +"_seed_" + str(seed)]) + env_mean
			env_min = np.minimum(np.array(data[env +"_seed_" + str(seed)]), env_min)
			env_max = np.maximum(np.array(data[env +"_seed_" + str(seed)]), env_min)
			
	env_mean = env_mean/settings.num_seeds
	env_mean = [sum(env_mean[i:i+smoothing_size])/smoothing_size for i in range(0,len(env_mean), smoothing_size)]

	x = [i*smoothing_size for i in range(len(env_mean))]
	x_all = [i for i in range(len(env_max))]
	plt.plot(x[:-1], env_mean[:-1], color = colors[env_num])
	plt.fill_between(x_all, env_max, env_min, color = colors[env_num], alpha=0.13)
	plt.gca().legend(training_envs, loc="lower right")
plt.ylim((-800,0))
plt.savefig("training_envs.png")
plt.show()