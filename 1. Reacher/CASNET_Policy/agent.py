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
		self.seq_lens = envs.envs_seq_lens
		self.robot_encoder = nn.GRU(p.link_dims, p.encoded_robot_dims, batch_first=True)
		self.fc = nn.Sequential(
									nn.Linear(p.encoded_robot_dims + 2, p.fc1_dims),	#2 is the goal dimensions
									nn.Tanh(),
									nn.Linear(p.fc1_dims, p.fc2_dims),
									nn.Tanh(),
								)
		self.action_decoder = nn.GRU(p.encoded_robot_dims+p.fc2_dims, 1, batch_first=True)
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

		for name, param in self.robot_encoder.named_parameters():
			if "bias" in name:
				nn.init.constant_(param, 0.0)
			elif "weight" in name:
				nn.init.xavier_normal_(param)

		for name, param in self.action_decoder.named_parameters():
			if "bias" in name:
				nn.init.constant_(param, 0.0)
			elif "weight" in name:
				nn.init.xavier_normal_(param)

		nn.init.constant_(self.logstd, 0.0)

	def init_hidden_robot_encoder(self, batch_size):
		return torch.zeros(1,batch_size, p.encoded_robot_dims).cuda()

	def init_hidden_action_encoder(self, batch_size):
		return torch.zeros(1,batch_size, 1).cuda()

	def forward(self, obs, goals):
		goals = goals.cuda()
		obs = obs.view(obs.shape[0], settings.max_links, int(obs.shape[1]/settings.max_links)).cuda()
		batch_size = obs.shape[0]
		seq_lens = self.seq_lens*int(batch_size/len(self.envs.envs))
		robot_encoder_hidden = self.init_hidden_robot_encoder(batch_size)
		obs = pack_padded_sequence(obs, seq_lens, batch_first=True, enforce_sorted=False)
		encoded_robots_states, encoded_robots = self.robot_encoder(obs, robot_encoder_hidden)
		encoded_robots_states, _ = pad_packed_sequence(encoded_robots_states, batch_first=True)
		fc_input = torch.cat((goals, encoded_robots.view(encoded_robots.shape[1], encoded_robots.shape[2])), dim=1)
		fc_out = self.fc(fc_input)
		fc_out.shaped = fc_out.view(fc_out.shape[0], 1, fc_out.shape[1])
		decoder_input = torch.cat((encoded_robots_states, fc_out.shaped.repeat(1, encoded_robots_states.shape[1],1)), dim=2)
		mean_actions, _ = self.action_decoder(decoder_input)
		mean_actions = mean_actions.view(mean_actions.shape[0], mean_actions.shape[1])

		logstd = self.logstd[:encoded_robots_states.shape[1]]
		logstd = logstd.view(1,logstd.shape[0])
		logstd = logstd.repeat(batch_size, 1)
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