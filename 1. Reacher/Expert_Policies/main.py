import torch, random, datetime, os, datetime
import numpy as np

from parameters import settings
from parameters import hyperparameters as p

def set_seeds(seed = 1):
	torch.manual_seed(seed)
	random.seed(seed)
	np.random.seed(seed)

set_seeds()

import gym
from agent import Agent
from trainer import trainer

#Loading custom envs
import Reacher_10, Reacher_11, Reacher_12
import Reacher_20, Reacher_21, Reacher_22
import Reacher_30, Reacher_31, Reacher_32
import Reacher_40, Reacher_41, Reacher_42
import Reacher_50, Reacher_51, Reacher_52
import Reacher_60, Reacher_61, Reacher_62

def train_single_env(env, Agent):
	trainer(env, Agent)

def collect_data():
	_ = os.system("rm Reacher_expert_PPO_data.csv")
	t_start = datetime.datetime.now()
	for env_name in settings.env_names:
		for seed in range(settings.num_seeds):
			set_seeds(seed)
			env = gym.make(env_name)
			env.name = env_name
			trainer(env, Agent, store_data=True, seed=seed+1, t_start = t_start)


if __name__ == "__main__":
	if settings.mode == "train_single":
		env = gym.make(settings.env_names[0])
		env.name = settings.env_names[0]
		train_single_env(env, Agent)
	else:
		collect_data()
