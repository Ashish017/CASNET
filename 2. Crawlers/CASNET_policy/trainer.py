from envs import Envs
from agent import Policy, QNetwork
import time, itertools, torch, random, os
from logger import Logger

from torch.optim import Adam
import torch.nn.functional as F

from parameters import (hyperparameters as p, settings)

class ReplayBuffer:

	def __init__(self):
		self.device = torch.device(settings.device)
		self.capacity = p.buffer_capacity
		self.buffer = []
		self.filled = 0

		self.push_size = len(settings.env_names)

	def to_tensor(self, value):
		return torch.FloatTensor(value).to(self.device)

	def push(self, names, leg_starts, states, next_states, actions, rewards, dones):
		actions = torch.FloatTensor(actions).to(self.device)

		try:
			self.names = torch.cat((self.names, names))
			self.leg_starts = torch.cat((self.leg_starts, leg_starts))
			self.states = torch.cat((self.states, states))
			self.actions = torch.cat((self.actions, actions))
			self.rewards = torch.cat((self.rewards, rewards))
			self.next_states = torch.cat((self.next_states, next_states))
			self.dones = torch.cat((self.dones, dones))

		except AttributeError:
			self.names = names
			self.leg_starts = leg_starts
			self.states = states
			self.actions = actions
			self.rewards = rewards
			self.next_states = next_states
			self.dones = dones

		self.filled += self.push_size

		if self.filled > self.capacity:
			self.names, self.leg_starts, self.states, self.actions, self.rewards, self.next_states, self.dones = self.names[self.push_size:], self.leg_starts[self.push_size:], self.states[self.push_size:], self.actions[self.push_size:], self.rewards[self.push_size:], self.next_states[self.push_size:], self.dones[self.push_size:]
			self.filled -= self.push_size

	def sample(self):
		indexes = torch.LongTensor(random.sample(range(len(self.states)),p.batch_size)).to(self.device)
		sample_names = torch.index_select(self.names, 0, indexes, out=None)
		sample_leg_starts = torch.index_select(self.leg_starts, 0, indexes, out=None)
		sample_states = torch.index_select(self.states, 0, indexes, out=None)
		sample_next_states = torch.index_select(self.next_states, 0, indexes, out=None)
		sample_actions = torch.index_select(self.actions, 0, indexes, out=None)
		sample_rewards = torch.index_select(self.rewards, 0, indexes, out=None)
		sample_dones = torch.index_select(self.dones, 0, indexes, out=None)

		return sample_names, sample_leg_starts, sample_states, sample_actions, sample_rewards, sample_next_states, sample_dones

	def __len__(self):
		return self.filled

class Trainer:

	def __init__(self):
		#Preparing envs
		self.envs = Envs()

		self.memory = ReplayBuffer()
		self.device = torch.device(settings.device)
		self.policy = Policy().to(self.device)
		self.policy_optim = Adam(self.policy.parameters(), lr=p.lr)

		self.critic = QNetwork().to(self.device)
		self.critic_target = QNetwork().to(self.device)

		self.critic_optim = Adam(self.critic.parameters(), lr=p.lr)
		self.parameter_update(tau=1.0)

		if settings.mode == "test":
			self.policy.load_state_dict(torch.load("policy_seed_{}".format(settings.seed)))

		self.logger = Logger()

	def start(self):
		self.total_numsteps = 0

		if settings.mode == "train":
			self.add_random_steps()

			names = torch.FloatTensor([i for i,_ in enumerate(settings.env_names)]).to(self.device)
			while self.total_numsteps < p.max_numsteps:
				self.run_test()
				leg_starts, states = self.envs.reset() 
				for step in range(p._max_episode_steps):
					self.total_numsteps += 1
					actions = self.select_action(leg_starts, states, names)
					next_states, rewards, dones = self.envs.step(actions)
					self.memory.push(names, leg_starts, states, next_states, actions, rewards, dones)
					states = self.envs.reset_dones(next_states, dones)

					c1_loss, c2_loss, policy_loss = self.update_nets()
					
					if (self.total_numsteps%10) == 0:
						self.logger.show_update(self.total_numsteps)

			torch.save(self.policy.state_dict(), "policy_seed_{}".format(settings.seed))

		else:
			print("Seed: {}".format(settings.seed))
			self.run_test()

	def run_test(self):
		if settings.mode == "test":
			print("\nTesting current policy")
		leg_starts, states = self.envs.reset()
		done_filter = epsd_rewards = torch.FloatTensor([1.0]*len(settings.env_names)).to(self.device)
		epsd_rewards = torch.FloatTensor([0.0]*len(settings.env_names)).to(self.device)
		names = torch.FloatTensor([i for i,_ in enumerate(settings.env_names)]).to(self.device)
		for step in range(p._max_episode_steps):
			actions = self.select_action(leg_starts, states, names, evaluate=True)
			next_states, rewards, dones = self.envs.step(actions)
			epsd_rewards += done_filter*rewards
			done_filter *= (dones!=1).float()
			states = next_states
		
		self.logger.add_rewards(len(names), epsd_rewards, self.total_numsteps)
		self.logger.save()

	def add_random_steps(self):
		print("Adding random steps")
		leg_starts, states = self.envs.reset()
		names = torch.FloatTensor([i for i,_ in enumerate(settings.env_names)]).to(self.device)
		while len(self.memory) <= p.batch_size*10:
			actions = self.envs.sample_actions()
			next_states, rewards, dones = self.envs.step(actions)
			self.memory.push(names, leg_starts, states, next_states, actions, rewards, dones)
			states = self.envs.reset_dones(next_states, dones)

	def select_action(self, leg_starts, states, names, evaluate=False):
		with torch.no_grad():
						
			if not evaluate:
				actions, _, _ = self.policy.sample(leg_starts,states,names)
			else:
				_, _, actions = self.policy.sample(leg_starts,states,names)

			return actions.cpu()

	def parameter_update(self, tau=p.tau):
		for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
			target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

	def update_nets(self):
		names_batch, leg_starts_batch, state_batch, action_batch, reward_batch, next_state_batch, mask_batch = self.memory.sample()

		reward_batch = reward_batch.unsqueeze(1)
		mask_batch = mask_batch.unsqueeze(1)

		with torch.no_grad():
			next_state_action, next_state_log_pi, _ = self.policy.sample(leg_starts_batch, next_state_batch, names_batch)
			qf1_next_target, qf2_next_target = self.critic_target(leg_starts_batch, next_state_batch, next_state_action, names_batch)
			min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - p.alpha * next_state_log_pi
			next_q_value = reward_batch + mask_batch * p.gamma * (min_qf_next_target)
		qf1, qf2 = self.critic(leg_starts_batch, state_batch, action_batch, names_batch)

		qf1_loss = F.mse_loss(qf1, next_q_value)
		qf2_loss = F.mse_loss(qf2, next_q_value)
		qf_loss = qf1_loss + qf2_loss

		self.critic_optim.zero_grad()
		qf_loss.backward()
		self.critic_optim.step()

		pi, log_pi, _ = self.policy.sample(leg_starts_batch, state_batch, names_batch)
		qf1_pi, qf2_pi = self.critic(leg_starts_batch, state_batch, pi, names_batch)
		min_qf_pi = torch.min(qf1_pi, qf2_pi)
		policy_loss = ((p.alpha * log_pi) - min_qf_pi).mean()
		
		self.policy_optim.zero_grad()
		policy_loss.backward()
		self.policy_optim.step()

		self.parameter_update()

		return qf1_loss.item(), qf2_loss.item(), policy_loss.item()