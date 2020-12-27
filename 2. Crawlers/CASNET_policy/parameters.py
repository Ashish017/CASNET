class Settings:

	def __init__(self):
		self.device = "cuda"
		self.max_links = 3
		self.max_legs = 6
		
	def set_modeAndSeed(self, mode, seed):
		self.seed = seed
		self.mode = mode
		if mode == "train":
			self.env_names = ["Quadruped-v10","Quadruped-v11",\
						"Quadruped-v21","Quadruped-v22","Quadruped-v26",\
						"Quadruped-v32","Quadruped-v33",\
						"Quadruped-v43","Quadruped-v44",\
						"Hexapod-v10","Hexapod-v11","Hexapod-v16",\
						"Hexapod-v21","Hexapod-v22","Hexapod-v25",\
						"Hexapod-v32","Hexapod-v33",\
						"Hexapod-v43","Hexapod-v44",\
						]
		else:
			self.env_names = ["Quadruped-v12","Quadruped-v13","Quadruped-v14",\
						"Quadruped-v20","Quadruped-v23","Quadruped-v24","Quadruped-v25",\
						"Quadruped-v31","Quadruped-v34","Quadruped-v35",\
						"Quadruped-v41","Quadruped-v42","Quadruped-v45",\
						"Quadruped-v51","Quadruped-v52","Quadruped-v53","Quadruped-v54","Quadruped-v55",\
						"Hexapod-v12","Hexapod-v13","Hexapod-v14","Hexapod-v15",\
						"Hexapod-v20","Hexapod-v23","Hexapod-v24","Hexapod-v26",\
						"Hexapod-v31","Hexapod-v34","Hexapod-v35",\
						"Hexapod-v41","Hexapod-v42","Hexapod-v45",\
						"Hexapod-v51","Hexapod-v52","Hexapod-v53","Hexapod-v54","Hexapod-v55"\
						]
		self.num_envs = len(self.env_names)

class Hyperparameters:

	def __init__(self):
		self.max_numsteps = 200000
		self._max_episode_steps = 1000
		self.batch_size = 256
		self.random_steps = 1000
		self.buffer_capacity = 1.5 * int(1e6)

		self.leg_segment_dims = 7
		self.encoded_leg_dims = 32
		self.encoded_robot_dims = 256
		self.hidden_dims = 256

		self.epsilon = 1e-6
		self.lr = 3e-4
		self.tau = 0.05
		self.alpha = 0.2
		self.gamma = 0.99


settings = Settings()
hyperparameters = Hyperparameters()