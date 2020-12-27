class Parameters:

	def __init__(self):
		self.lr = 0.0001
		self.replay_k = 4
		self.buffer_size = 5*int(1e5)
		self.clip_range = 10.0

		self.n_epochs = 250
		self.n_cycles = 50
		self.update_per_episode = 40
		self.batch_size = 64
		self.hidden_dims = 256
		self.max_episode_steps = 50

		self.noise_eps = 0.2
		self.random_eps = 0.3

		self.clip_obs = 200.0
		self.polyak = 0.95
		self.gamma = 0.98

		self.eps = 0.01

		self.testing_eps = 10

		self.device = "cuda"

hyperparameters = Parameters()