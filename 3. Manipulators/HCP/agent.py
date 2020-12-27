import torch, time
import torch.nn as nn
import torch.nn.functional as F
from parameters import hyperparameters as p

class Actor(nn.Module):

	def __init__(self):
		super(Actor, self).__init__()

		self.device = torch.device(p.device)

		self.fc = nn.Sequential(
									nn.Linear((p.max_dof-1)*p.link_dims + p.goal_dims, p.hidden_dims),	#+3 for goal dims
									nn.ReLU(),
									nn.Linear(p.hidden_dims, p.hidden_dims),
									nn.ReLU(),
									nn.Linear(p.hidden_dims, p.hidden_dims),
									nn.ReLU(),
								)
		self.action_layer = nn.Linear(p.hidden_dims, 7)
		self.init_weight()

	def init_weight(self):
		for m in self.fc:
			if hasattr(m, 'weight') or hasattr(m, 'bias'):
				for name, param in m.named_parameters():
					if name == "weight":
						nn.init.orthogonal_(param, gain=nn.init.calculate_gain('relu'))
					if name == "bias":
						nn.init.constant_(param, 0.0)

	def add_extra(self, actions):
		zeros = torch.zeros(*actions.shape[:-1], 1).to(self.device)
		actions = torch.cat((actions, zeros), 1)
		return actions

	def forward(self, obs, goals):
		obs = torch.cat((obs, goals), 1)

		fc_out = self.fc(obs)
		actions = self.action_layer(fc_out)
		
		return actions

class Critic(nn.Module):

	def __init__(self):
		super(Critic, self).__init__()

		self.device = torch.device(p.device)
		
		self.fc = nn.Sequential(
									nn.Linear((p.max_dof-1)*p.link_dims + p.goal_dims + (p.max_dof-1), p.hidden_dims),	#obs, goals, and actions
									nn.ReLU(),
									nn.Linear(p.hidden_dims, p.hidden_dims),
									nn.ReLU(),
									nn.Linear(p.hidden_dims, p.hidden_dims),
									nn.ReLU(),
								)
		self.value_layer = nn.Linear(p.hidden_dims, 1)
		
		self.init_weight()

	def init_weight(self):
		for m in self.fc:
			if hasattr(m, 'weight') or hasattr(m, 'bias'):
				for name, param in m.named_parameters():
					if name == "weight":
						nn.init.orthogonal_(param, gain=nn.init.calculate_gain('relu'))
					if name == "bias":
						nn.init.constant_(param, 0.0)

	def forward(self, obs, goals, actions):
		actions = actions[:,:-1]
		obs = torch.cat((obs, goals, actions), 1)

		fc_out = self.fc(obs)
		values = self.value_layer(fc_out)
		values = values.view(values.shape[0])
		
		return values
