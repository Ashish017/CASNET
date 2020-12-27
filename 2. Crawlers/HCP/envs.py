import gym, time
from parameters import settings
import torch
import numpy as np

#Loading custom envs
import Quadruped_10,Quadruped_11,Quadruped_12,Quadruped_13,Quadruped_14
import Quadruped_20,Quadruped_21,Quadruped_22,Quadruped_23,Quadruped_24,Quadruped_25,Quadruped_26
import Quadruped_31,Quadruped_32,Quadruped_33,Quadruped_34,Quadruped_35
import Quadruped_41,Quadruped_42,Quadruped_43,Quadruped_44,Quadruped_45
import Quadruped_51,Quadruped_52,Quadruped_53,Quadruped_54,Quadruped_55

import Hexapod_10,Hexapod_11,Hexapod_12,Hexapod_13,Hexapod_14, Hexapod_15, Hexapod_16
import Hexapod_20,Hexapod_21,Hexapod_22,Hexapod_23,Hexapod_24,Hexapod_25,Hexapod_26
import Hexapod_31,Hexapod_32,Hexapod_33,Hexapod_34,Hexapod_35
import Hexapod_41,Hexapod_42,Hexapod_43,Hexapod_44,Hexapod_45
import Hexapod_51,Hexapod_52,Hexapod_53,Hexapod_54,Hexapod_55

class Model_database():

	def __init__(self):
		self.device = torch.device(settings.device)
		self.use_index_dict = {
					'Hexapod-v10': [0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16],
					'Hexapod-v11': [0, 1, 3, 4, 5, 6, 7, 9, 10, 12, 13, 15, 16],
					'Hexapod-v12': [0, 1, 3, 4, 5, 6, 7, 9, 10, 12, 13, 15, 16, 17],
					'Hexapod-v13': [0, 1, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 15, 16, 17],
					'Hexapod-v14': [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 15, 16, 17],
					'Hexapod-v15': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17],
					'Hexapod-v16': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
					'Hexapod-v20': [0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16],
					'Hexapod-v21': [0, 1, 3, 4, 5, 6, 7, 9, 10, 12, 13, 15, 16],
					'Hexapod-v22': [0, 1, 3, 4, 5, 6, 7, 9, 10, 12, 13, 15, 16, 17],
					'Hexapod-v23': [0, 1, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 15, 16, 17],
					'Hexapod-v24': [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 15, 16, 17],
					'Hexapod-v25': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17],
					'Hexapod-v26': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
					'Hexapod-v31': [0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16],
					'Hexapod-v32': [0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16],
					'Hexapod-v33': [0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16],
					'Hexapod-v34': [0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16],
					'Hexapod-v35': [0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16],
					'Hexapod-v41': [0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16],
					'Hexapod-v42': [0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16],
					'Hexapod-v43': [0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16],
					'Hexapod-v44': [0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16],
					'Hexapod-v45': [0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16],
					'Hexapod-v51': [0, 1, 3, 4, 5, 6, 7, 9, 10, 12, 13, 15, 16, 17],
					'Hexapod-v52': [0, 1, 3, 4, 5, 6, 7, 9, 10, 12, 13, 15, 16],
					'Hexapod-v53': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
					'Hexapod-v54': [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16],
					'Hexapod-v55': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
					'Quadruped-v10': [0, 1, 3, 4, 6, 7, 9, 10],
					'Quadruped-v11': [0, 1, 2, 3, 4, 6, 7, 9, 10],
					'Quadruped-v12': [0, 1, 2, 3, 4, 5, 6, 7, 9, 10],
					'Quadruped-v13': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
					'Quadruped-v14': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
					'Quadruped-v20': [0, 1, 3, 4, 6, 7, 9, 10],
					'Quadruped-v21': [0, 1, 2, 3, 4, 6, 7, 9, 10],
					'Quadruped-v22': [0, 1, 3, 4, 5, 6, 7, 9, 10],
					'Quadruped-v23': [0, 1, 3, 4, 6, 7, 8, 9, 10],
					'Quadruped-v24': [0, 1, 3, 4, 6, 7, 9, 10, 11],
					'Quadruped-v25': [0, 1, 3, 4, 5, 6, 7, 9, 10, 11],
					'Quadruped-v26': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
					'Quadruped-v31': [0, 1, 3, 4, 6, 7, 9, 10],
					'Quadruped-v32': [0, 1, 3, 4, 6, 7, 9, 10],
					'Quadruped-v33': [0, 1, 3, 4, 6, 7, 9, 10],
					'Quadruped-v34': [0, 1, 3, 4, 6, 7, 9, 10],
					'Quadruped-v35': [0, 1, 3, 4, 6, 7, 9, 10],
					'Quadruped-v41': [0, 1, 3, 4, 6, 7, 9, 10],
					'Quadruped-v42': [0, 1, 3, 4, 6, 7, 9, 10],
					'Quadruped-v43': [0, 1, 3, 4, 6, 7, 9, 10],
					'Quadruped-v44': [0, 1, 3, 4, 6, 7, 9, 10],
					'Quadruped-v45': [0, 1, 3, 4, 6, 7, 9, 10],
					'Quadruped-v51': [0, 1, 3, 4, 6, 7, 9, 10],
					'Quadruped-v52': [0, 1, 3, 4, 6, 7, 8, 9, 10],
					'Quadruped-v53': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
					'Quadruped-v54': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
					'Quadruped-v55': [0, 1, 3, 4, 5, 6, 7, 9, 10, 11]
		}

		self.non_padded_leg_seq_dict = {}
		for env_name in self.use_index_dict.keys():
			seq = []
			num_legs = 4 if env_name[0] == "Q" else 6
			for i in range(2,18,3):
				if i not in self.use_index_dict[env_name]:
					if i > 11 and num_legs == 4:
						seq.append(3)
					elif i > 11 and num_legs == 6:
						seq.append(2)
					else:
						seq.append(2)
				else:
					seq.append(3)
			self.non_padded_leg_seq_dict[env_name] = seq

	def get_action(self, env_name, action):
		return np.take(action, self.use_index_dict[env_name], out=None)

	def create_non_padded_leg_seq_len(self, name_index):
		seq = []
		for num in name_index:
			env = settings.env_names[int(num)]
			seq += self.non_padded_leg_seq_dict[env]
		return seq

	def create_non_padded_robot_seq_len(self, name_index):
		seq = []
		for num in name_index:
			env = settings.env_names[int(num)]
			seq.append(4 if env[0] == "Q" else 6)
		return seq

database = Model_database()

class Envs():
	def __init__(self):
		self.database = Model_database()
		self.num_envs = len(settings.env_names)
		self.action_dims = settings.max_legs*settings.max_links
		self.device = torch.device(settings.device)
		self.envs = []
		for name in settings.env_names:
			env = gym.make(name, add_dummy=True)
			env.seed(settings.seed)
			env.name = name
			self.envs.append(env)

	def _shaped_obs(self, obs):
		obs = np.array(obs)
		obs = obs.reshape(settings.max_legs, obs.shape[0]//settings.max_legs)
		leg_locations = obs[:, :2]
		obs = obs[:, 2:]
		return leg_locations, obs

	def reset(self):
		obs = []
		leg_locations = []
		for env in self.envs:
			ob = env.reset()
			leg_location, ob = self._shaped_obs(ob)
			obs.append(ob)
			leg_locations.append(leg_location)
		leg_locations = torch.FloatTensor(leg_locations).to(self.device)
		obs = torch.FloatTensor(obs).to(self.device)
		return leg_locations, obs

	def step(self, actions):
		next_states, rewards, dones = [],[],[]
		for i, env in enumerate(self.envs):
			next_state, reward, done, _ = env.step(self.database.get_action(env.name, actions[i]))
			_, next_state = self._shaped_obs(next_state)
			next_states.append(next_state)
			rewards.append(reward)
			dones.append(done)
		next_states = torch.FloatTensor(next_states).to(self.device)
		rewards = torch.FloatTensor(rewards).to(self.device)
		dones = torch.FloatTensor(dones).to(self.device)
		return next_states, rewards, dones

	def reset_dones(self, states, dones):
		done_indexes = torch.nonzero(dones).view(-1)
		for index in done_indexes:
			state = self.envs[index].reset()
			_, state = self._shaped_obs(state)
			states[index] = torch.FloatTensor(state).to(self.device)
		return states

	def sample_actions(self):
		return np.random.rand(self.num_envs, self.action_dims)*2-1

class RegularizedEnv():
	def __init__(self, env_name):
		self.name = env_name
		self.env = gym.make(env_name, add_dummy=True)
		self.env.seed(settings.seed)
		self.env.name = self.name
		self.device = torch.device(settings.device)

	def _shaped_obs(self, obs):
		obs = torch.FloatTensor(obs)
		obs = obs.view(settings.max_legs, obs.shape[0]//settings.max_legs)
		leg_locations = obs[:, :2]
		obs = obs[:, 2:]
		return leg_locations, obs

	def reset(self):
		obs = self.env.reset()
		return (self._shaped_obs(obs))

	def step(self, action):
		action = database.get_action(self.name, action)
		obs, reward, done, _ = self.env.step(action)
		_, obs = self._shaped_obs(obs)
		return obs, reward, done

	def sample_action(self):
		return torch.FloatTensor(settings.max_links*settings.max_legs).uniform_(-1,1)