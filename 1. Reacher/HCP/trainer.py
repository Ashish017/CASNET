import torch
import torch.optim as optim

import pandas as pd
import numpy as np
import datetime, os, random

from parameters import hyperparameters as p
from parameters import settings	

def manager(agent, envs):
	agent.eval()
	goals, obs = envs.reset()
	dones = torch.FloatTensor([0 for _ in obs])
	
	for idx in range(p.N_steps):
		actions, values, neglogpacs = agent.step(obs, goals)

		try:
			mb_obs = torch.cat((mb_obs, obs.view(1, obs.shape[0], obs.shape[1])))
			mb_goals = torch.cat((mb_goals, goals.view(1, goals.shape[0], goals.shape[1])))
			mb_actions = torch.cat((mb_actions, actions.view(1, actions.shape[0], actions.shape[1])))
			mb_values = torch.cat((mb_values, values.view(1, values.shape[0])))
			mb_neglogpacs = torch.cat((mb_neglogpacs, neglogpacs.view(1, neglogpacs.shape[0])))
			mb_dones = torch.cat((mb_dones, dones.view(1, dones.shape[0])))

		except NameError:
			mb_obs = obs.view(1, obs.shape[0], obs.shape[1])
			mb_goals = goals.view(1, goals.shape[0], goals.shape[1])
			mb_actions = actions.view(1, actions.shape[0], actions.shape[1])
			mb_values = values.view(1, values.shape[0])
			mb_neglogpacs = neglogpacs.view(1, neglogpacs.shape[0])
			mb_dones = dones.view(1, dones.shape[0])

		goals, obs, rewards, dones = envs.step(actions)
		try:
			mb_rewards = torch.cat((mb_rewards, rewards.view(1, rewards.shape[0])))
		except NameError:
			mb_rewards = rewards.view(1, rewards.shape[0])

		if idx == p.N_steps-1:
			_, last_values, _ = agent.step(obs, goals)

	mb_advs = torch.zeros_like(mb_rewards)
	mb_returns = mb_advs

	#Calculating GAE
	lastgaelam = torch.zeros_like(mb_values[0])
	for t in reversed(range(p.N_steps)):
		if t == p.N_steps - 1:
			nextnonterminal = 1.0 - mb_dones[-1]
			nextvalue = last_values
		else:
			nextnonterminal = 1.0 - mb_dones[t+1]
			nextvalue = mb_values[t+1]
		delta = mb_rewards[t] + (p.gamma*nextvalue*nextnonterminal) - mb_values[t]
		mb_advs[t] = lastgaelam = delta + (p.gamma*p.lam*nextnonterminal*lastgaelam)
	mb_returns = mb_advs + mb_values
	
	return mb_obs, mb_actions, mb_goals, mb_values, mb_neglogpacs, mb_returns, mb_rewards

def trainer(envs, agent, seed):

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

	if isinstance(p.lr, float): lr=constfn(p.lr)
	if isinstance(p.cliprange, float): cliprange = constfn(p.cliprange)

	if settings.mode not in ["train", "train_6"]:
		p.N_updates = 500

	elif settings.mode == "train_6":
		p.N_updates = 150

	t_start = datetime.datetime.now()
	for update in range(p.N_updates):
		obs, actions, goals, values, neglogpacs, returns, rewards = manager(agent, envs)
		
		rewards = rewards.sum(dim=0)
		rewards = rewards

		if not seed:
			if not update%5:
				os.system("clear")
				percent = round((100*(update+1)/p.N_updates),2)
				done = "|"*int(percent/2)
				remaining = "-" * (50 - len(done))
				eta = (100-percent) * ((datetime.datetime.now()-t_start)/percent)
				print("Mode: " + settings.mode + "\tETA: {}".format(str(eta).split(".")[0]))
				print("|" + done + remaining + "| {}%".format(round(percent, 2)))

		if not update:
			all_rewards = rewards.view(1, rewards.shape[0])
		else:
			all_rewards = torch.cat((all_rewards, rewards.view(1, rewards.shape[0])))

		if settings.mode in ["train", "train_6"]:

			frac = 1.0 - (update+1)/p.N_updates
			lrnow = lr(frac)
			cliprangenow = cliprange(frac)
			optimizer = optim.Adam(agent.parameters(), lr=lrnow)

			agent.train()
			inds = np.arange(p.N_steps)
			np.random.shuffle(inds)

			for start in range(0, p.N_steps, p.batch_size):
				end = start + p.batch_size

				obs_, goals_, returns_, actions_, values_, neglogpacs_ = indexSelect([obs, goals, returns, actions, values, neglogpacs], start, end)
				advs_ = returns_ - values_
				advs_ = (advs_ - advs_.mean()) / (advs_.std() + 1e-8)

				optimizer.zero_grad()
				neglogp, entropy, vpred = agent.statistics(obs_, goals_ , actions_)
				entropy = torch.mean(entropy, dim=-2)
				vpred_clip = values_ + torch.clamp(vpred - values_, -cliprangenow, cliprangenow)

				vf_loss = torch.max((vpred - returns_) ** 2, (vpred_clip - returns_) ** 2)
				vf_loss = 0.5 * torch.mean(vf_loss)

				ratio = torch.exp(neglogpacs_ - neglogp)
				pg_loss = torch.max(- advs_ * ratio, - advs_ * torch.clamp(ratio, 1.0-cliprangenow, 1.0+cliprangenow))
				pg_loss = torch.mean(pg_loss)

				loss = (pg_loss - entropy * p.entropy_coef + vf_loss * p.vf_coef).sum()/len(envs.envs)
				loss.backward()
				optimizer.step()

	if settings.mode == "train":
		torch.save(agent.state_dict(), "seed_"+str(seed))

	elif settings.mode == "train_6":
		torch.save(agent.state_dict(), "_6x_seed_"+str(seed))

	di = {}
	for i in range(len(envs.envs)):
		env = envs.envs[i]
		di[env.name + "_seed_{}".format(seed)] = all_rewards[:,i]
	frame = pd.DataFrame(di)
	return frame