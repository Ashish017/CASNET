import math

class Settings:

	def __init__(self):
		self.num_seeds = 3

		self.training_envs = ["Reacher-v10", "Reacher-v20","Reacher-v30", "Reacher-v40","Reacher-v50"]
		self.testing_envs = ["Reacher-v11", "Reacher-v12","Reacher-v21", "Reacher-v22","Reacher-v31", "Reacher-v32","Reacher-v41", "Reacher-v42","Reacher-v51", "Reacher-v52", "Reacher-v60","Reacher-v61","Reacher-v62"]
		self.train_env_6 = ["Reacher-v60"]

		self.max_links = 6	#Variable just to make our life easier
		self.save_freq = 100

		self.max_velocity = 0.1
		self.max_joint_pos = math.pi * 2
		self.max_link_length = 0.2

	def set_mode(self, mode):
		self.mode = mode

		if self.mode in ["train"]:
			self.env_names = self.training_envs
		elif self.mode == "train_6":
			self.env_names = self.train_env_6
		else:
			self.env_names = self.testing_envs

class Hyperparameters:

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

		self.link_dims = 3
		self.encoded_robot_dims = 64
		self.fc1_dims = 64
		self.fc2_dims = 64

settings = Settings()
hyperparameters = Hyperparameters()