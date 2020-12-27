import random
import numpy as np
import time, torch

class ReplayMemory:
    def __init__(self, capacity, seed, cuda):
        random.seed(seed)
        self.device = torch.device("cuda") if cuda else torch.device("cpu") 
        self.capacity = capacity
        self.buffer = []
        self.filled = 0
        self.position = 0

    def to_tensor(self, value):
        return torch.FloatTensor(value).to(self.device)

    def push(self, state, action, reward, next_state, done):
        state, action, reward, next_state, done = self.to_tensor([state]),self.to_tensor([action]),self.to_tensor([reward]),self.to_tensor([next_state]),self.to_tensor([done])
        self.filled += 1
        if self.filled <= self.capacity:
            try:
                self.states = torch.cat((self.states, state))
                self.actions = torch.cat((self.actions, action))
                self.rewards = torch.cat((self.rewards, reward))
                self.next_states = torch.cat((self.next_states, next_state))
                self.dones = torch.cat((self.dones, done))

            except AttributeError:
                self.states = state
                self.actions = action
                self.rewards = reward
                self.next_states = next_state
                self.dones = done

        else:
            self.states[self.position] = state
            self.actions[self.position] = action
            self.rewards[self.position] = reward
            self.next_states[self.position] = next_state
            self.dones[self.position] = done

            self.filled = self.capacity
            self.position += 1
            self.position = self.position%self.capacity

    def sample(self, batch_size):
        indexes = torch.LongTensor(random.sample(range(len(self.states)),batch_size)).to(self.device)
        sample_states = torch.index_select(self.states, 0, indexes, out=None)
        sample_next_states = torch.index_select(self.next_states, 0, indexes, out=None)
        sample_actions = torch.index_select(self.actions, 0, indexes, out=None)
        sample_rewards = torch.index_select(self.rewards, 0, indexes, out=None)
        sample_dones = torch.index_select(self.dones, 0, indexes, out=None)

        return sample_states, sample_actions, sample_rewards, sample_next_states, sample_dones

    def __len__(self):
        return self.filled
