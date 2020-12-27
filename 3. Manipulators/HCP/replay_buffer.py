from parameters import hyperparameters as p
import time
import numpy as np
import pickle

class her_sampler:
	def __init__(self):
		self.future_p = 1 - (1. / (1 + p.replay_k))
		self.distance_threshold = 0.05

	def compute_reward(self, achieved, goal):
		assert goal.shape == achieved.shape
		d = np.linalg.norm(goal - achieved, axis=-1)
		return -(d > self.distance_threshold).astype(np.float32)

	def sample_her_transitions(self, episode_batch, batch_size_in_transitions):
		T = episode_batch['actions'].shape[1]
		rollout_batch_size = episode_batch['actions'].shape[0]
		batch_size = batch_size_in_transitions
		# select which rollouts and which timesteps to be used
		episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
		t_samples = np.random.randint(T, size=batch_size)
		transitions = {key: episode_batch[key][episode_idxs, t_samples].copy() for key in episode_batch.keys()}
		# her idx
		her_indexes = np.where(np.random.uniform(size=batch_size) < self.future_p)
		future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
		future_offset = future_offset.astype(int)
		future_t = (t_samples + future_offset)[her_indexes]
		# replace go with achieved goal
		future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]
		transitions['g'][her_indexes] = future_ag
		# to get the params to re-compute reward
		transitions['r'] = np.expand_dims(self.compute_reward(transitions['ag_next'], transitions['g']), 1)
		transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:]) for k in transitions.keys()}

		return transitions

class normalizer:
	def __init__(self, size):
		self.size = size
		# some local information
		self.total_sum = np.zeros(self.size, np.float32)
		self.total_sumsq = np.zeros(self.size, np.float32)
		self.total_count = np.ones(1, np.float32)
		# get the mean and std
		self.mean = np.ones(self.size, np.float32)
		self.std = np.ones(self.size, np.float32)
		
	# update the parameters of the normalizer
	def update(self, v):
		v = v.reshape(-1, self.size)
		# do the computing
		self.total_sum += v.sum(axis=0)
		self.total_sumsq += (np.square(v)).sum(axis=0)
		self.total_count += v.shape[0]
		
		self.mean = self.total_sum / self.total_count
		self.std = np.sqrt(np.maximum(np.square(p.eps), (self.total_sumsq / self.total_count) - np.square(self.total_sum / self.total_count)))
	
	# normalize the observation
	def normalize(self, v):
		prev_shape = v.shape
		v = v.reshape(-1, self.size)
		v = np.clip((v - self.mean) / (self.std), -p.clip_range, p.clip_range)
		v = v.reshape(*prev_shape)
		return v

class replay_buffer:
	def __init__(self, seed):
		self.seed = seed
		self.T = p.max_episode_steps
		self.size = p.buffer_size // self.T
		# memory management
		self.current_size = 0
		self.n_transitions_stored = 0
		# create the buffer to store info
		self.buffers = {'obs': np.empty([self.size, self.T, p.max_dof-1, p.link_dims]),
						'ag': np.empty([self.size, self.T, 3]),
						'g': np.empty([self.size, self.T, 3]),
						'obs_next': np.empty([self.size, self.T,  p.max_dof-1, p.link_dims]),
						'ag_next': np.empty([self.size, self.T, 3]),
						'actions': np.empty([self.size, self.T, p.max_dof]),
						'seq': np.empty([self.size, self.T])
						}
		self.o_norm = normalizer(p.link_dims *(p.max_dof-1))
		self.g_norm = normalizer(p.goal_dims)

		self.her_module = her_sampler()

	def store_episode(self, episode_batch):
		mb_obs, mb_ag, mb_g, mb_obs_next, mb_ag_next, mb_actions, mb_seq = episode_batch
		batch_size = mb_obs.shape[0]
		
		idxs = self._get_storage_idx()
		self.buffers['obs'][idxs] = mb_obs
		self.buffers['ag'][idxs] = mb_ag
		self.buffers['g'][idxs] = mb_g
		self.buffers['obs_next'][idxs] = mb_obs_next
		self.buffers['ag_next'][idxs] = mb_ag_next
		self.buffers['actions'][idxs] = mb_actions
		self.buffers['seq'][idxs] = mb_seq
		self.n_transitions_stored += self.T * batch_size

	def save_normalizers(self):
		with open("o_norm_seed_{}".format(self.seed), 'wb') as file:
			pickle.dump(self.o_norm, file)
		with open("g_norm_seed_{}".format(self.seed), 'wb') as file:
			pickle.dump(self.g_norm, file)

	def load_normalizers(self):
		with open('o_norm_seed_{}'.format(self.seed), "rb") as file:
			self.o_norm = pickle.load(file)
		with open("g_norm_seed_{}".format(self.seed), "rb") as file:
			self.g_norm = pickle.load(file)

	# sample the data from the replay buffer
	def sample(self):
		temp_buffers = {}
		for key in self.buffers.keys():
			temp_buffers[key] = self.buffers[key][:self.current_size]
		# sample transitions
		transitions = self.her_module.sample_her_transitions(temp_buffers, p.batch_size)
		return transitions

	def _get_storage_idx(self):
		if self.current_size+p.batch_size <= self.size:
			idx = np.arange(self.current_size, self.current_size+p.batch_size)
		elif self.current_size < self.size:
			overflow = p.batch_size - (self.size - self.current_size)
			idx_a = np.arange(self.current_size, self.size)
			idx_b = np.random.randint(0, self.current_size, overflow)
			idx = np.concatenate([idx_a, idx_b])
		else:
			idx = np.random.randint(0, self.size, p.batch_size)
		self.current_size = min(self.size, self.current_size+p.batch_size)
		return idx

	def update_normalizer(self, episode_batch):
		mb_obs, mb_ag, mb_g, mb_obs_next, mb_ag_next, mb_actions, mb_seq = episode_batch
		# get the number of normalization transitions
		num_transitions = mb_actions.shape[1]
		# create the new buffer to store them
		buffer_temp = {'obs': mb_obs, 
					   'ag': mb_ag,
					   'g': mb_g, 
					   'actions': mb_actions, 
					   'obs_next': mb_obs_next,
					   'ag_next': mb_ag_next,
					   'seq': mb_seq
					   }
		transitions = self.her_module.sample_her_transitions(buffer_temp, num_transitions)
		obs, g = transitions['obs'], transitions['g']
		# pre process the obs and g
		transitions['obs'], transitions['g'] = self.preproc_og(obs, g)
		# update
		self.o_norm.update(transitions['obs'])
		self.g_norm.update(transitions['g'])

	def preproc_og(self, o, g):
		o = np.clip(o, -p.clip_obs, p.clip_obs)
		g = np.clip(g, -p.clip_obs, p.clip_obs)
		return o, g
