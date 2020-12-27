import torch, time
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from parameters import (hyperparameters as p, settings)
from envs import Model_database

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

database = Model_database()

def weights_init_(m):
	if isinstance(m, nn.Linear):
		torch.nn.init.xavier_uniform_(m.weight, gain=1)
		torch.nn.init.constant_(m.bias, 0)

class Policy(nn.Module):
	def __init__(self):
		super().__init__()
		self.device = torch.device(settings.device)

		self.leg_encoder = nn.GRU(p.leg_segment_dims, p.encoded_leg_dims, batch_first=True)
		self.robot_encoder = nn.GRU(p.encoded_leg_dims+2, p.encoded_robot_dims, batch_first=True)

		self.fc = nn.Sequential(
									nn.Linear(p.encoded_robot_dims, p.hidden_dims),
									nn.ReLU(),
									nn.Linear(p.hidden_dims, p.hidden_dims),
									nn.ReLU(),
								)

		self.legs_decoder = nn.GRU(p.hidden_dims+p.encoded_robot_dims, p.encoded_leg_dims, batch_first=True)
		self.actions_decoder = nn.GRU(p.encoded_leg_dims+p.encoded_leg_dims, 1, batch_first=True)

		self.logstd = nn.Linear(p.encoded_leg_dims+p.encoded_leg_dims, 1)

		self.init_weights()

	def init_weights(self):
		for m in self.fc:
			if hasattr(m, 'weight') or hasattr(m, 'bias'):
				for name, param in m.named_parameters():
					if name == "weight":
						nn.init.xavier_uniform_(param, gain=1)
					if name == "bias":
						nn.init.constant_(param, 0.0)

	def init_hidden_leg_encoder(self, batch_size):
		encoder_hidden = torch.FloatTensor(1, batch_size, p.encoded_leg_dims).zero_()
		return encoder_hidden.to(self.device)

	def init_hidden_robot_encoder(self, batch_size):
		encoder_hidden = torch.FloatTensor(1,batch_size, p.encoded_robot_dims).zero_()
		return encoder_hidden.to(self.device)

	def init_hidden_leg_decoder(self, batch_size):
		decoder_hidden = torch.FloatTensor(1, batch_size, p.encoded_leg_dims).zero_()
		return decoder_hidden.to(self.device)

	def init_hidden_actions_decoder(self, batch_size):
		decoder_hidden = torch.FloatTensor(1, batch_size, 1).zero_()
		return decoder_hidden.to(self.device)

	def forward(self, leg_starts, states, names):
		num_robots = states.shape[0]
		states = states.view(states.shape[0]*states.shape[1],states.shape[2]//p.leg_segment_dims, p.leg_segment_dims)
		num_legs = states.shape[0]

		non_padded_leg_seq_len = database.create_non_padded_leg_seq_len(names.cpu().numpy())
		packed_leg_seq = pack_padded_sequence(states, non_padded_leg_seq_len, batch_first=True, enforce_sorted=False)
		leg_encoder_hidden = self.init_hidden_leg_encoder(num_legs)
		leg_encoder_states, encoded_legs = self.leg_encoder(packed_leg_seq, leg_encoder_hidden)
		leg_encoder_states, _ = pad_packed_sequence(leg_encoder_states, batch_first=True, total_length=settings.max_links)
		encoded_legs = encoded_legs.view(num_robots, settings.max_legs, p.encoded_leg_dims)

		non_padded_robot_seq_len = database.create_non_padded_robot_seq_len(names.cpu().numpy())
		robot_encoder_input = torch.cat((leg_starts, encoded_legs), dim=2)
		robot_encoder_hidden = self.init_hidden_robot_encoder(num_robots)
		packed_robot_seq = pack_padded_sequence(robot_encoder_input, non_padded_robot_seq_len,batch_first=True, enforce_sorted=False)
		robot_encoder_states, encoded_robots = self.robot_encoder(packed_robot_seq, robot_encoder_hidden)
		robot_encoder_states, _ = pad_packed_sequence(robot_encoder_states, batch_first=True,total_length=settings.max_legs)

		encoded_robots = self.fc(encoded_robots)
		encoded_robots = encoded_robots.view(encoded_robots.shape[1], encoded_robots.shape[0], encoded_robots.shape[2])

		encoded_robots_repeated = encoded_robots.repeat(1, robot_encoder_states.shape[1], 1)
		leg_decoder_input = torch.cat((encoded_robots_repeated, robot_encoder_states), dim=2)
		leg_decoder_hidden = self.init_hidden_leg_decoder(num_robots)
		decoded_legs, _ = self.legs_decoder(leg_decoder_input, leg_decoder_hidden)

		decoded_legs_repeated = decoded_legs.contiguous().view(decoded_legs.shape[0]*decoded_legs.shape[1],1, decoded_legs.shape[2]).repeat(1, leg_encoder_states.shape[1], 1)
		actions_decoder_input = torch.cat((decoded_legs_repeated, leg_encoder_states), dim=2)
		actions_decoder_hidden = self.init_hidden_actions_decoder(actions_decoder_input.shape[0])
		actions, _ = self.actions_decoder(actions_decoder_input, actions_decoder_hidden)
		mean_actions = actions.contiguous().view(num_robots, settings.max_legs*settings.max_links)

		logstd = self.logstd(actions_decoder_input)
		logstd = logstd.view(num_robots, settings.max_legs*settings.max_links)

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

		self.leg_encoder1 = nn.GRU(p.leg_segment_dims, p.encoded_leg_dims, batch_first=True)
		self.robot_encoder1 = nn.GRU(p.encoded_leg_dims+2, p.encoded_robot_dims, batch_first=True)

		self.fc1 = nn.Sequential(
									nn.Linear(p.encoded_robot_dims, p.hidden_dims),
									nn.ReLU(),
									nn.Linear(p.hidden_dims, p.hidden_dims),
									nn.ReLU(),
									nn.Linear(p.hidden_dims, 1)
								)
		self.leg_encoder2 = nn.GRU(p.leg_segment_dims, p.encoded_leg_dims, batch_first=True)
		self.robot_encoder2 = nn.GRU(p.encoded_leg_dims+2, p.encoded_robot_dims, batch_first=True)

		self.fc2 = nn.Sequential(
									nn.Linear(p.encoded_robot_dims, p.hidden_dims),
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

	def init_hidden_leg_encoder(self, batch_size):
		encoder_hidden = torch.FloatTensor(1, batch_size, p.encoded_leg_dims).zero_()
		return encoder_hidden.to(self.device)

	def init_hidden_robot_encoder(self, batch_size):
		encoder_hidden = torch.FloatTensor(1,batch_size, p.encoded_robot_dims).zero_()
		return encoder_hidden.to(self.device)

	def forward(self, leg_starts, states, actions, names, q=False):
		num_robots = states.shape[0]
		states = states.view(states.shape[0]*states.shape[1],states.shape[2]//p.leg_segment_dims, p.leg_segment_dims)
		num_legs = states.shape[0]

		non_padded_leg_seq_len = database.create_non_padded_leg_seq_len(names.cpu().numpy())
		packed_leg_seq = pack_padded_sequence(states, non_padded_leg_seq_len, batch_first=True, enforce_sorted=False)
		leg_encoder_hidden = self.init_hidden_leg_encoder(num_legs)
		leg_encoder_states, encoded_legs = self.leg_encoder1(packed_leg_seq, leg_encoder_hidden)
		leg_encoder_states, _ = pad_packed_sequence(leg_encoder_states, batch_first=True, total_length=settings.max_links)
		encoded_legs = encoded_legs.view(num_robots, settings.max_legs, p.encoded_leg_dims)

		non_padded_robot_seq_len = database.create_non_padded_robot_seq_len(names.cpu().numpy())
		robot_encoder_input = torch.cat((leg_starts, encoded_legs), dim=2)
		robot_encoder_hidden = self.init_hidden_robot_encoder(num_robots)
		packed_robot_seq = pack_padded_sequence(robot_encoder_input, non_padded_robot_seq_len,batch_first=True, enforce_sorted=False)
		robot_encoder_states, encoded_robots = self.robot_encoder1(packed_robot_seq, robot_encoder_hidden)
		
		encoded_robots = encoded_robots.view(encoded_robots.shape[1], encoded_robots.shape[2])
		fc_input = torch.cat((encoded_robots, actions), dim=1)

		Q1 = self.fc1(encoded_robots)

		non_padded_leg_seq_len = database.create_non_padded_leg_seq_len(names.cpu().numpy())
		packed_leg_seq = pack_padded_sequence(states, non_padded_leg_seq_len, batch_first=True, enforce_sorted=False)
		leg_encoder_hidden = self.init_hidden_leg_encoder(num_legs)
		leg_encoder_states, encoded_legs = self.leg_encoder2(packed_leg_seq, leg_encoder_hidden)
		leg_encoder_states, _ = pad_packed_sequence(leg_encoder_states, batch_first=True, total_length=settings.max_links)
		encoded_legs = encoded_legs.view(num_robots, settings.max_legs, p.encoded_leg_dims)

		non_padded_robot_seq_len = database.create_non_padded_robot_seq_len(names.cpu().numpy())
		robot_encoder_input = torch.cat((leg_starts, encoded_legs), dim=2)
		robot_encoder_hidden = self.init_hidden_robot_encoder(num_robots)
		packed_robot_seq = pack_padded_sequence(robot_encoder_input, non_padded_robot_seq_len,batch_first=True, enforce_sorted=False)
		robot_encoder_states, encoded_robots = self.robot_encoder2(packed_robot_seq, robot_encoder_hidden)
		
		encoded_robots = encoded_robots.view(encoded_robots.shape[1], encoded_robots.shape[2])
		fc_input = torch.cat((encoded_robots, actions), dim=1)
		
		Q2 = self.fc2(encoded_robots)
		
		return Q1, Q2