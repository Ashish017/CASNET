import torch
import numpy as np
import torch.nn as nn
import random
from torch.distributions.normal import Normal

from parameters import hyperparameters as p
from parameters import settings

class Agent(nn.Module):

	def __init__(self, env):
		super(Agent, self).__init__()

		self.fc = nn.Sequential(
									nn.Linear(env.observation_space.shape[0], 64),
									nn.Tanh(),
									nn.Linear(64, 64),
									nn.Tanh(),
								)
		self.action_layer = nn.Linear(64, env.action_space.shape[0])
		self.value_layer = nn.Linear(64, 1)
		self.logstd = nn.Parameter(torch.zeros(env.action_space.shape[0]))
		
		self.init_weights()

	def init_weights(self):
		for m in self.fc:
			if hasattr(m, 'weight') or hasattr(m, 'bias'):
				for name, param in m.named_parameters():
					if name == "weight":
						nn.init.orthogonal_(param, gain=nn.init.calculate_gain('tanh'))
					if name == "bias":
						nn.init.constant_(param, 0.0)

		for name, param in self.value_layer.named_parameters():
			if name == "weight":
				nn.init.orthogonal_(param, gain=0.01)
			if name == "bias":
				nn.init.constant_(param, 0.0)

		for name, param in self.action_layer.named_parameters():
			if name == "weight":
				nn.init.orthogonal_(param, gain=0.01)
			if name == "bias":
				nn.init.constant_(param, 0.0)

		nn.init.constant_(self.logstd, 0.0)

	def forward(self, obs):
		fc_out = self.fc(obs)
		self.mean_actions = self.action_layer(fc_out)
		log_std = self.logstd
		std = torch.exp(log_std)
		self.pd = Normal(self.mean_actions, std)
		self.v = self.value_layer(fc_out)

	def step(self, obs):
		with torch.no_grad():
			self.forward(obs)
			act = self.pd.sample()
			neglogp = torch.sum(-self.pd.log_prob(act)).view(1)
		return act, self.v, neglogp

	def statistics(self, obs, actions):
		self.forward(obs)
		neglogps = torch.sum(-self.pd.log_prob(actions), dim=1)
		entropies = torch.sum(self.pd.entropy(), dim=1)
		values = self.v
		neglogps = neglogps.view(obs.shape[0],1)
		entropies = torch.mean(entropies).view(1)

		return neglogps, entropies, values