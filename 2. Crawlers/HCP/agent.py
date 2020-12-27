import torch, time
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from parameters import (hyperparameters as p, settings)
from envs import Model_database

database = Model_database()

def weights_init_(m):
	if isinstance(m, nn.Linear):
		torch.nn.init.xavier_uniform_(m.weight, gain=1)
		torch.nn.init.constant_(m.bias, 0)

class Policy(nn.Module):
	def __init__(self):
		super().__init__()
		self.device = torch.device(settings.device)

		self.fc = nn.Sequential(
									nn.Linear(126+12, p.hidden_dims),
									nn.ReLU(),
									nn.Linear(p.hidden_dims, p.hidden_dims),
									nn.ReLU(),
								)

		self.action_layer = nn.Linear(p.hidden_dims, settings.max_legs*settings.max_links)
		self.logstd = nn.Linear(p.hidden_dims, settings.max_legs*settings.max_links)

		self.init_weights()

	def init_weights(self):
		for m in self.fc:
			if hasattr(m, 'weight') or hasattr(m, 'bias'):
				for name, param in m.named_parameters():
					if name == "weight":
						nn.init.orthogonal_(param, gain=nn.init.calculate_gain('tanh'))
					if name == "bias":
						nn.init.constant_(param, 0.0)

	def forward(self, leg_starts, states, names):
		states = states.view(states.shape[0], states.shape[1]*states.shape[2])
		leg_starts = leg_starts.view(leg_starts.shape[0], leg_starts.shape[1]*leg_starts.shape[2])
		states = torch.cat((states, leg_starts),1)

		fc_out = self.fc(states)
		mean_actions = self.action_layer(fc_out)
		logstd = self.logstd(fc_out)

		return mean_actions, logstd

	def sample(self, leg_starts, states, names):
		mean, logstd = self.forward(leg_starts, states, names)
		std = logstd.exp()
		normal = Normal(mean, std)
		x_t = normal.rsample()
		y_t = torch.tanh(x_t)
		action = y_t
		log_prob = normal.log_prob(x_t)
		log_prob -= torch.log(1 - y_t.pow(2) + p.epsilon)
		log_prob = log_prob.sum(1, keepdim=True)
		mean = torch.tanh(mean)
		return action, log_prob, mean

class QNetwork(nn.Module):
	def __init__(self):
		super().__init__()
		self.device = torch.device(settings.device)

		self.fc1 = nn.Sequential(
									nn.Linear(126+12+18, p.hidden_dims),
									nn.ReLU(),
									nn.Linear(p.hidden_dims, p.hidden_dims),
									nn.ReLU(),
									nn.Linear(p.hidden_dims, 1)
								)
		
		self.fc2 = nn.Sequential(
									nn.Linear(126+12+18, p.hidden_dims),
									nn.ReLU(),
									nn.Linear(p.hidden_dims, p.hidden_dims),
									nn.ReLU(),
									nn.Linear(p.hidden_dims, 1)
								)
		self.init_weights()

	def init_weights(self):
		for m in self.fc1:
			if hasattr(m, 'weight') or hasattr(m, 'bias'):
				for name, param in m.named_parameters():
					if name == "weight":
						nn.init.xavier_uniform_(param, gain=1)
					if name == "bias":
						nn.init.constant_(param, 0.0)

		for m in self.fc2:
			if hasattr(m, 'weight') or hasattr(m, 'bias'):
				for name, param in m.named_parameters():
					if name == "weight":
						nn.init.xavier_uniform_(param, gain=1)
					if name == "bias":
						nn.init.constant_(param, 0.0)

	def forward(self, leg_starts, states, actions, names, q=False):
		states = states.view(states.shape[0], states.shape[1]*states.shape[2])
		leg_starts = leg_starts.view(leg_starts.shape[0], leg_starts.shape[1]*leg_starts.shape[2])
		states = torch.cat((states, leg_starts, actions),1)

		Q1 = self.fc1(states)
		Q2 = self.fc2(states)

		return Q1, Q2