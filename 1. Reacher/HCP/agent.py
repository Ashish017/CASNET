import torch, random, time
from torch.distributions.normal import Normal
import torch.nn as nn

from parameters import hyperparameters as p
from parameters import settings

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class Agent(nn.Module):

	def __init__(self, envs):
		super(Agent, self).__init__()

		self.envs = envs
		self.fc = nn.Sequential(
									nn.Linear((settings.max_links * p.link_dims) + 2, p.fc1_dims),	#2 is the goal dimensions
									nn.Tanh(),
									nn.Linear(p.fc1_dims, p.fc2_dims),
									nn.Tanh(),
								)
		self.action_layer = nn.Linear(p.fc2_dims, settings.max_links)
		self.value_layer = nn.Linear(p.fc2_dims, 1)
		self.logstd = nn.Parameter(torch.zeros(settings.max_links))

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

	def forward(self, obs, goals):
		batch_size = obs.shape[0]
		fc_input = torch.cat((obs, goals), 1)
		fc_out = self.fc(fc_input)

		mean_actions = self.action_layer(fc_out)
		logstd = self.logstd.view(1, self.logstd.shape[0])
		logstd = self.logstd.repeat(batch_size, 1)
		std = torch.exp(logstd)

		self.pd = Normal(mean_actions, std)
		self.v = self.value_layer(fc_out).view(-1)

	def step(self, obs, goals):
		with torch.no_grad():
			self.forward(obs, goals)
			act = self.pd.sample()
			neglogp = torch.sum(-self.pd.log_prob(act), dim=1)
		return act.cpu(), self.v.cpu(), neglogp.cpu()

	def statistics(self, obs, goals, actions):
		obs = obs.view(obs.shape[0]*obs.shape[1], obs.shape[2])
		goals = goals.view(goals.shape[0]*goals.shape[1], goals.shape[2])
		actions = actions.view(actions.shape[0]*actions.shape[1], actions.shape[2])
		
		self.forward(obs, goals)

		neglogps = torch.sum(-self.pd.log_prob(actions), dim=1)
		entropies = torch.sum(self.pd.entropy(), dim=1)
		values = self.v

		neglogps = neglogps.view(p.batch_size, int(neglogps.shape[0]/p.batch_size))
		entropies = entropies.view(p.batch_size, int(entropies.shape[0]/p.batch_size))
		values = values.view(p.batch_size, int(values.shape[0]/p.batch_size))
		
		return neglogps, entropies, values