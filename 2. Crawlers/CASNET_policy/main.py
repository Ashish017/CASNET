import time, random, os, torch
import numpy as np
from parameters import (hyperparameters as p, settings)

def set_seed(seed=1):
	torch.manual_seed(seed)
	random.seed(seed)
	np.random.seed(seed)

set_seed()

from trainer import Trainer

modes = ["train", "test"]
seeds = [3]

for seed in seeds:
	for mode in modes:
		set_seed(seed)
		settings.set_modeAndSeed(mode, seed)
		trainer = Trainer()
		trainer.start()