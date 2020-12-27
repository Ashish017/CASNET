import gym, time
from parameters import settings
import torch
import numpy as np

class Load_envs():

	def __init__(self, add_dummy=True):
		self.envs = []
		self.envs_seq_lens = []
		for env_name in settings.env_names:
			env = gym.make(env_name, add_dummy=add_dummy, max_links = settings.max_links)
			env.name = env_name
			self.envs.append(env)
			self.envs_seq_lens.append(int(env_name[-2]))

	def reset(self):
		obs = []
		for env in self.envs:
			ob = env.reset()
			obs.append(ob)
		obs =  torch.FloatTensor(obs)
		return obs[:,:2], obs[:, 2:]

	def step(self, actions):
		goals, obs, rewards, dones = [], [], [], []
		for i in range(len(self.envs)):
			env = self.envs[i]
			action = actions[i][:int(env.name[-2])]
			ob, reward, done, _ = env.step(action)
			goals.append(ob[0:2])
			obs.append(ob[2:])
			rewards.append(reward)
			dones.append(done)
		goals = torch.FloatTensor(goals)
		obs = torch.FloatTensor(obs)
		rewards = torch.FloatTensor(rewards)
		dones = torch.FloatTensor(dones)

		return goals, obs, rewards, dones