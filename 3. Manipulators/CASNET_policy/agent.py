import torch, time
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from parameters import hyperparameters as p

class Actor(nn.Module):

	def __init__(self):
		super(Actor, self).__init__()

		self.device = torch.device(p.device)

		self.robot_encoder = nn.GRU(p.link_dims, p.encoded_robot_dims, batch_first=True)
		self.fc = nn.Sequential(
									nn.Linear(p.encoded_robot_dims + 3, p.hidden_dims),	#+3 for goal dims
									nn.ReLU(),
									nn.Linear(p.hidden_dims, p.hidden_dims),
									nn.ReLU(),
								)
		self.action_decoder = nn.GRU(p.encoded_robot_dims+p.hidden_dims, 1, batch_first=True)
		self.init_weight()

	def init_weight(self):
		for m in self.fc:
			if hasattr(m, 'weight') or hasattr(m, 'bias'):
				for name, param in m.named_parameters():
					if name == "weight":
						nn.init.orthogonal_(param, gain=nn.init.calculate_gain('relu'))
					if name == "bias":
						nn.init.constant_(param, 0.0)

	def init_hidden_robot_encoder(self, num_robots):
		return torch.zeros(1,num_robots, p.encoded_robot_dims).to(self.device)
	def init_hidden_action_encoder(self, num_robots):
		return torch.zeros(1, num_robots, 1).to(self.device)

	def add_extra(self, actions):
		zeros = torch.zeros(*actions.shape[:-1], 1).to(self.device)
		actions = torch.cat((actions, zeros), 1)
		return actions

	def forward(self, obs, goals, seq):
		num_robots = obs.shape[0]
		
		robot_encoder_hidden = self.init_hidden_robot_encoder(num_robots)
		packed_obs = pack_padded_sequence(obs, seq, batch_first=True, enforce_sorted=False)
		encoded_robot_states, encoded_robots = self.robot_encoder(packed_obs, robot_encoder_hidden)
		encoded_robot_states, _ = pad_packed_sequence(encoded_robot_states, batch_first = True, total_length=p.max_dof-1)
		encoded_robots = encoded_robots.view(encoded_robots.shape[1], encoded_robots.shape[2])
		fc_input = torch.cat((encoded_robots, goals), dim=1)
		fc_out = self.fc(fc_input)
		
		fc_out_shaped = (fc_out.view(fc_out.shape[0], 1, fc_out.shape[1])).repeat(1, encoded_robot_states.shape[1], 1)
		decoder_input = torch.cat((encoded_robot_states, fc_out_shaped), dim=2)
		
		actions, _ = self.action_decoder(decoder_input)
		actions = torch.tanh(actions.view(actions.shape[0], actions.shape[1]))
		actions = self.add_extra(actions)

		return actions

class Critic(nn.Module):

	def __init__(self):
		super(Critic, self).__init__()

		self.device = torch.device(p.device)
		
		self.robot_encoder = nn.GRU(p.link_dims, p.encoded_robot_dims, batch_first=True)
		self.fc = nn.Sequential(
									nn.Linear(p.encoded_robot_dims + 3 + p.max_dof-1, p.hidden_dims),	#+3 for goal dims + 6 for action dims
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

	def init_hidden_robot_encoder(self, batch_size):
		return torch.ones(1,batch_size, p.encoded_robot_dims).to(self.device)

	def forward(self, obs, goals, actions, seq):
		actions = actions[:,:-1]
		if obs.shape[0] == p.max_dof-1:
			obs = obs.unsqueeze(0)
			goals = goals.unsqueeze(0)
		else:
			obs = obs.view(p.batch_size, p.max_dof-1, p.link_dims)

		num_robots = obs.shape[0]
		robot_encoder_hidden = self.init_hidden_robot_encoder(num_robots)
		packed_obs = pack_padded_sequence(obs, seq, batch_first=True, enforce_sorted=False)
		encoded_robot_states, encoded_robots = self.robot_encoder(packed_obs, robot_encoder_hidden)
		encoded_robot_states, _ = pad_packed_sequence(encoded_robot_states, batch_first = True, total_length=p.max_dof-1)
		encoded_robots = encoded_robots.view(encoded_robots.shape[1], encoded_robots.shape[2])
		fc_input = torch.cat((encoded_robots, goals, actions), dim=1)
		fc_out = self.fc(fc_input)
		values = self.value_layer(fc_out).view(num_robots)

		return values
