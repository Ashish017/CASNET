import torch
import torch.nn as nn
import torch.nn.functional as F
from parameters import hyperparameters as p

class Actor(nn.Module):
	def __init__(self, ob_shape, goal_shape, action_space):
		super(Actor, self).__init__()
		self.fc = nn.Sequential(
									nn.Linear(ob_shape + goal_shape, p.hidden_dims),
									nn.ReLU(),
									nn.Linear(p.hidden_dims, p.hidden_dims),
									nn.ReLU(),
								)
		
		self.action_out = nn.Linear(p.hidden_dims, action_space)

	def init_weight(self):
		for m in self.fc:
			if hasattr(m, 'weight') or hasattr(m, 'bias'):
				for name, param in m.named_parameters():
					if name == "weight":
						nn.init.orthogonal_(param, gain=nn.init.calculate_gain('relu'))
					if name == "bias":
						nn.init.constant_(param, 0.0)

	def forward(self, obs, goal):
		try:
			x = torch.cat((obs, goal))
		except RuntimeError:
			x = torch.cat((obs, goal), 1)
		
		x = self.fc(x)
		actions = torch.tanh(self.action_out(x))
		return actions

class Critic(nn.Module):
	def __init__(self, ob_shape, goal_shape, action_space):
		super(Critic, self).__init__()
		self.fc = nn.Sequential(
									nn.Linear(ob_shape + goal_shape + action_space, p.hidden_dims),
									nn.ReLU(),
									nn.Linear(p.hidden_dims, p.hidden_dims),
									nn.ReLU(),
								)
		
		self.q_out = nn.Linear(p.hidden_dims, action_space)

	def init_weight(self):
		for m in self.fc:
			if hasattr(m, 'weight') or hasattr(m, 'bias'):
				for name, param in m.named_parameters():
					if name == "weight":
						nn.init.orthogonal_(param, gain=nn.init.calculate_gain('relu'))
					if name == "bias":
						nn.init.constant_(param, 0.0)

	def forward(self, obs, goal, actions):
		try:
			x = torch.cat((obs, goal, actions))
		except RuntimeError:
			x = torch.cat((obs, goal, actions), 1)
		x = self.fc(x)
		q_value = self.q_out(x)

		return q_value
