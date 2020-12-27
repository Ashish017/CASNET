import numpy as np
import random, time
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

	#time.sleep((seed-1)*30)

	if p.mode == 'train':
		envs_generator = Envs()
		training_envs, testing_envs = envs_generator.get_envs()
		set_seed(seed)
		trainer = Trainer(training_envs, testing_envs, seed)
		trainer.start()
	else:
		for variance_limit in p.variance_limits:
			envs_generator = Envs(variance_limit=variance_limit)
			training_envs = envs_generator.get_envs()
			testing_envs = training_envs
			set_seed(seed)
			trainer = Trainer(training_envs, testing_envs, seed, variance_limit = variance_limit)
			trainer.start()

seeds = [1,2,3]

if __name__ == '__main__':
	p = Pool(processes = len(seeds))
	_ = p.map(start, seeds)
	p.terminate()