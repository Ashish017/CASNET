import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import os, datetime
from parameters import hyperparameters as p
from parameters import settings

def manager(agent, env):

	"""
	Runs a single episode and returns the generated episode data and GAE
	"""

	agent.eval()
	obs = env.reset()
	obs = torch.FloatTensor(obs)

	for idx in range(p.N_steps):
		actions, values, neglogpacs = agent.step(obs)
		
		if idx == 0:
			mb_obs, mb_actions, mb_values, mb_neglogpacs = obs.view(1 ,obs.shape[0]), actions.view(1 ,actions.shape[0]), values.view(1 ,values.shape[0]), neglogpacs.view(1 ,neglogpacs.shape[0])
		else:
			mb_obs = torch.cat((mb_obs, obs.view(1 ,obs.shape[0])))
			mb_actions = torch.cat((mb_actions, actions.view(1 ,actions.shape[0])))
			mb_values = torch.cat((mb_values, values.view(1 ,values.shape[0])))
			mb_neglogpacs = torch.cat((mb_neglogpacs, neglogpacs.view(1 ,neglogpacs.shape[0])))
		
		obs, rewards, done, _ = env.step(actions.cpu())
		obs = torch.FloatTensor(obs)
		done = torch.tensor(done).float()

		if idx == 0:
			mb_rewards = rewards.view(1,1)
			mb_dones = done.view(1,1)
		else:
			mb_rewards = torch.cat((mb_rewards, rewards.view(1,1)))
			mb_dones = torch.cat((mb_dones, done.view(1,1)))

		if idx == p.N_steps-1:
			_, last_value, _ = agent.step(obs)

	#Now calculating GAE
	mb_advs = torch.zeros_like(mb_values)
	lastgaelam = 0

	for t in reversed(range(p.N_steps)):
		if t == p.N_steps-1:
			nextnonterminal = 1.0 - mb_dones[-1]
			nextvalue = last_value
		else:
			nextnonterminal = 1.0 - mb_dones[t+1]
			nextvalue = mb_values[t+1]
		delta = mb_rewards[t] + (p.gamma*nextvalue*nextnonterminal) - mb_values[t]
		mb_advs[t] = lastgaelam = delta + (p.gamma*p.lam*nextnonterminal*lastgaelam)
	
	mb_returns = mb_advs + mb_values

	return mb_obs, mb_actions, mb_values, mb_neglogpacs, mb_returns, mb_rewards

def trainer(env, Agent, store_data = False, seed = 1, t_start=datetime.datetime.now()):

	"""
	Implements the PPO algorithm
	"""
	
	def constfn(val):
		def f(_):
			return val
		return f

	def indexSelect(objects, start, end):
		indexes = torch.LongTensor(inds[start:end])
		temp = []
		for Object in objects:
			temp.append(torch.index_select(Object, 0, indexes, out=None))
		return temp

	def show_update(t_start, env, seed, update):
		total_updates = len(settings.env_names) * settings.num_seeds * p.N_updates
		done = (settings.env_names.index(env.name) * settings.num_seeds * p.N_updates) + (seed-1)*p.N_updates + update
		percent = (done/total_updates)*100
		t_elapsed = (datetime.datetime.now() - t_start)
		t_remaining = (t_elapsed*(total_updates-done)/done)

		os.system("clear")
		print("Env: " + str(env.name) + " seed: " + str(seed) + " update: " + str(update) + "/" + str(p.N_updates))
		print("Time_elapsed: " + str(t_elapsed).split(".")[0])
		print("Estimated total time remaining: " + str(t_remaining).split(".")[0])
		print("|" + "#"*int(percent/2) + "-"*int((100-percent)/2) + "| " + str(round(percent,3)) + "%")

	agent = Agent(env)
	all_rewards = []
	if isinstance(p.lr, float): lr=constfn(p.lr)
	if isinstance(p.cliprange, float): cliprange = constfn(p.cliprange)

	for update in range(p.N_updates):
		if not (update+1)%20: show_update(t_start, env, seed, update+1)			
		obs, actions, values, neglogpacs, returns, rewards = manager(agent, env)
		frac = 1.0-(update+1)/p.N_updates
		lrnow = lr(frac)
		cliprangenow = cliprange(frac)
		optimizer = optim.Adam(agent.parameters(), lr=lrnow)
		agent.train()
		inds = np.arange(p.N_steps)
		np.random.shuffle(inds)

		if store_data:
			all_rewards.append(rewards.sum().item())
			if update == p.N_updates-1:
				try:
					data = pd.read_csv("Reacher_expert_PPO_data.csv")
					data.insert(len(data.keys()),str(env.name) + "_seed_" + str(seed), all_rewards)
					data.to_csv("Generated_data/Reacher_expert_PPO_data.csv", index=False)

				except FileNotFoundError:
					data = {}
					data[str(env.name) + "_seed_" + str(seed)] = all_rewards
					data = pd.DataFrame(data)
					data.to_csv("Generated_data/Reacher_expert_PPO_data.csv", index=False)

		for _ in range(p.train_epochs):
			for start in range(0, p.N_steps, p.batch_size):
				end = start + p.batch_size
				obs_, returns_, actions_, values_, neglogpacs_ = indexSelect([obs, returns, actions, values, neglogpacs], start, end)
				advs_ = returns_ - values_
				advs_ = (advs_ - advs_.mean())/(advs_.std() + 1e-8)
				
				optimizer.zero_grad()
				neglogp, entropy, vpred = agent.statistics(obs_, actions_)
				vpred_clip = values_ + torch.clamp(vpred - values_, -cliprangenow, cliprangenow)
				vf_loss = torch.max((vpred - returns_) ** 2, (vpred_clip - returns_)**2)
				vf_loss = 0.5*torch.mean(vf_loss)
				
				ratio = torch.exp(neglogpacs_ - neglogp)
				pg_loss = torch.max(-advs_ * ratio, -advs_ * torch.clamp(ratio, 1.0-cliprangenow, 1.0+cliprangenow))
				pg_loss = torch.mean(pg_loss)

				loss = (pg_loss - entropy * p.entropy_coef + vf_loss * p.vf_coef).sum()
				loss.backward()
				optimizer.step()

		if not (update+1)%settings.save_freq:
			torch.save(agent.state_dict(), env.name+"_"+str(seed))