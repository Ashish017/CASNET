import numpy as np
from trainer import Trainer
import random, time
import torch
from parameters import hyperparameters as p

def launch(name, seed):
	env = gym.make(name)
	env.ob_shape = env.observation_space['observation'].shape[0]
	env.goal_shape = env.observation_space['desired_goal'].shape[0]
	env.action_shape = env.action_space.shape[0]
	env.name = name
	
	env.seed(seed)
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if p.device == "cuda":
		torch.cuda.manual_seed(seed)
	
	trainer = Trainer(env, seed)
	trainer.start()

import numpy as np
import random
import torch

from multiprocessing import Pool

def set_seed(seed=1):
	torch.manual_seed(seed)
	random.seed(seed)
	np.random.seed(seed)

def start(seed):
	from parameters import hyperparameters as p
	from create_envs import Envs
	from trainer import Trainer
	envs_generator = Envs()
	training_envs, testing_envs = envs_generator.get_envs()
	set_seed(seed)		#Set_seed should be after Envs()
	for env in testing_envs.envs:
		env.ob_shape = env.observation_space['observation'].shape[0]
		env.goal_shape = env.observation_space['desired_goal'].shape[0]
		env.action_shape = env.action_space.shape[0]
		trainer = Trainer(env, seed)
		trainer.start()

seeds = [1,2,3]

if __name__ == '__main__':
	p = Pool(processes = len(seeds))
	_ = p.map(start, seeds)
	p.terminate()