class Parameters:

	def __init__(self):
		self.lr = 0.0001
		self.replay_k = 4
		self.buffer_size = int(1e6)
		self.clip_range = 10.0

		self.n_epochs = 250
		self.n_cycles = 50
		self.update_per_episode = 40
		self.batch_size = 64
		self.max_episode_steps = 50

		self.noise_eps = 0.2
		self.random_eps = 0.3

		self.clip_obs = 200.0
		self.polyak = 0.95
		self.gamma = 0.98

		self.eps = 0.01
		self.testing_eps = 10

		self.device = "cuda"
		self.max_dof = 7

		self.link_dims = 9
		self.goal_dims = 3
		self.hidden_dims = 256
		self.encoded_robot_dims = 256

		self.mode = 'train'	#or 'retrain'
		self.variance_limits = [0.3,0.4,0.5,0.6]

hyperparameters = Parameters()