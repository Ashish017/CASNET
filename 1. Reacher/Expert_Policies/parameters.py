import math

class Settings:

	"""
	Class to store settings
	"""

	def __init__(self):
		self.mode = "collect_data"	#Available modes: train_single / collect_data
		self.num_seeds = 3
		self.env_names = ["Reacher-v10","Reacher-v11","Reacher-v12",
							"Reacher-v20","Reacher-v21","Reacher-v22",
							"Reacher-v30","Reacher-v31","Reacher-v32",
							"Reacher-v40","Reacher-v41","Reacher-v42",
							"Reacher-v50","Reacher-v51","Reacher-v52",
							"Reacher-v60","Reacher-v61","Reacher-v62"
							]
		self.save_freq = 100

class Hyperparameters:
	"""
	Class to store hyperparameters
	"""

	def __init__(self):
		self.N_steps = 300
		self.N_updates = 2000
		self.batch_size = 4
		self.train_epochs = 1
		self.lr = 3e-4
		self.cliprange = 0.2
		self.gamma = 0.99
		self.lam = 0.95
		self.vf_coef = 0.5
		self.entropy_coef = 0.0

settings = Settings()
hyperparameters = Hyperparameters()