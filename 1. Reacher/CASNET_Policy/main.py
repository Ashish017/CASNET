import torch, random, time, os
import numpy as np

from parameters import settings
from parameters import hyperparameters as p

def set_seeds(seed=1):
	torch.manual_seed(seed)
	random.seed(seed)
	np.random.seed(seed)

set_seeds()

import gym, time
from load_envs import Load_envs
from agent import Agent
from trainer import trainer

#Loading custom envs
import Reacher_10, Reacher_11, Reacher_12
import Reacher_20, Reacher_21, Reacher_22
import Reacher_30, Reacher_31, Reacher_32
import Reacher_40, Reacher_41, Reacher_42
import Reacher_50, Reacher_51, Reacher_52
import Reacher_60, Reacher_61, Reacher_62

import multiprocessing
from multiprocessing import Pool

import pandas as pd

def run(seed):
	set_seeds(seed)
	envs = Load_envs()
	agent = Agent(envs).cuda()
	if settings.mode in ["test", "train_6"]:
		agent.load_state_dict(torch.load("seed_{}".format(seed)))
	elif settings.mode == "test_6":
		agent.load_state_dict(torch.load("_6x_seed_{}".format(seed)))
	seed_data = trainer(envs, agent, seed)
	return seed_data

if __name__ == "__main__":
	modes = ["train", "test", "base", "train_6", "test_6"]
	for mode in modes:
		settings.set_mode(mode)
		p = Pool(processes=settings.num_seeds)
		data = p.map(run, range(settings.num_seeds))
		p.terminate()
		data = pd.concat(data, axis=1)
		data.to_csv("Generated_data/Reacher_CASNET_PPO_" + mode + ".csv")