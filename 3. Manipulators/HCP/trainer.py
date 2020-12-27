import torch
import os, time
from datetime import datetime
import numpy as np
from replay_buffer import replay_buffer
from agent import Actor, Critic
import pandas as pd
from parameters import hyperparameters as p

class Trainer:
    def __init__(self, envs, testing_envs, seed, variance_limit=0.25):
        self.seed = seed
        self.successes = []
        self.testing_envs = testing_envs
        self.envs = envs
        self.variance_limit = variance_limit
        
        training_envs_per_dof = int(len(self.envs.envs)/3)
        
        self.training_env_seq = [4]*training_envs_per_dof + [5]*training_envs_per_dof + [6]*training_envs_per_dof
        self.testing_env_seq = [4]*10 + [5]*10 + [6]*10

        if p.mode == "retrain":
            self.training_env_seq = self.testing_env_seq

        self.device = torch.device(p.device)
        # create the network
        self.actor = Actor().to(self.device)
        self.critic = Critic().to(self.device)

        if p.mode == 'retrain':
            self.actor.load_state_dict(torch.load("actor_seed_{}".format(seed)))
            self.critic.load_state_dict(torch.load("critic_seed_{}".format(seed)))

        # build up the target network
        self.actor_target = Actor().to(self.device)
        self.critic_target = Critic().to(self.device)
        # load the weights into the target networks
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        # if use gpu
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=p.lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=p.lr)
        # her sampler
        self.buffer = replay_buffer(seed)

        if p.mode == 'retrain':
            self.buffer.load_normalizers()
            print("loading done")

        self.training_data, self.testing_data = {}, {}
        for env in self.envs.envs:
            self.training_data[env.name] = []
        for env in self.testing_envs.envs:
            self.testing_data[env.name] = []

        try:
            os.mkdir("Generated_data")
        except FileExistsError:
            pass
        
    def start(self):
        if p.mode == "retrain":
            for self.epoch in range(-10, 0):
                training_success_rate, testing_success_rate = self.eval_agent()     
                self.log_data(training_success_rate, testing_success_rate)

        else:
            for self.epoch in range(p.n_epochs):
                for _ in range(p.n_cycles):
                    # reset the rollouts
                    ep_obs, ep_ag, ep_g, ep_obs_next, ep_ag_next, ep_actions, ep_seq = [], [], [], [], [], [], []
                    # reset the environment
                    observation = self.envs.reset()
                    obs = observation['observation']
                    ag = observation['achieved_goal']
                    g = observation['desired_goal']

                    # start to collect samples
                    for t in range(p.max_episode_steps):
                        with torch.no_grad():
                            obs_norm, g_norm = self.normalize(obs, g)
                            pi = self.actor(obs_norm, g_norm)
                            action = self.add_noise(pi)
                          
                        # feed the actions into the environment
                        observation_new, info = self.envs.step(action)
                        obs_new = observation_new['observation']
                        ag_new = observation_new['achieved_goal']
                        # append rollouts
                        ep_obs.append(obs.copy())
                        ep_ag.append(ag.copy())
                        ep_g.append(g.copy())
                        ep_obs_next.append(obs_new.copy())
                        ep_ag_next.append(ag_new.copy())
                        ep_actions.append(action.copy())
                        ep_seq.append(self.training_env_seq)
                        # re-assign the observation
                        obs = obs_new
                        ag = ag_new

                    #convert them into arrays
                    ep_obs = np.array(ep_obs).swapaxes(0,1)
                    ep_ag = np.array(ep_ag).swapaxes(0,1)
                    ep_g = np.array(ep_g).swapaxes(0,1)
                    ep_obs_next = np.array(ep_obs_next).swapaxes(0,1)
                    ep_ag_next = np.array(ep_ag_next).swapaxes(0,1)
                    ep_actions = np.array(ep_actions).swapaxes(0,1)
                    ep_seq = np.array(ep_seq).swapaxes(0,1)

                    for i in range(ep_obs.shape[0]):
                        # store the episodes
                        self.buffer.store_episode([np.expand_dims(ep_obs[i],0), np.expand_dims(ep_ag[i],0), np.expand_dims(ep_g[i],0), np.expand_dims(ep_obs_next[i],0), np.expand_dims(ep_ag_next[i],0), np.expand_dims(ep_actions[i],0), np.expand_dims(ep_seq[i],0)])
                        self.buffer.update_normalizer([np.expand_dims(ep_obs[i],0), np.expand_dims(ep_ag[i],0), np.expand_dims(ep_g[i],0), np.expand_dims(ep_obs_next[i],0), np.expand_dims(ep_ag_next[i],0), np.expand_dims(ep_actions[i],0), np.expand_dims(ep_seq[i],0)])
                   
                    for _ in range(p.update_per_episode):
                        # train the network
                        c_loss, a_loss = self.update_network()
                        
                    # soft update
                    self.soft_update_target_network()

                training_success_rate, testing_success_rate = self.eval_agent()
                self.log_data(training_success_rate, testing_success_rate)
            
                torch.save(self.actor.state_dict(), "actor_seed_{}".format(self.seed))
                torch.save(self.critic.state_dict(), "critic_seed_{}".format(self.seed))
                self.buffer.save_normalizers()

    def log_data(self, training_data, testing_data):
        os.system("clear")
        print("Epoch: {}".format(self.epoch))
        print("Training_data: ")
        end = "\t"

        for i, env in enumerate(self.envs.envs):
            print(env.name, training_data[i], end=end)
            self.training_data[env.name].append(training_data[i])
            end = "\t" if end=="\n" else "\n"
        print(end="\n\n")
        
        frame = pd.DataFrame(self.training_data)
        if self.variance_limit == 0.25:
            frame.to_csv("Generated_data/" + p.mode + "ing_data_{}.csv".format(self.seed))
        else:
            frame.to_csv("Generated_data/" + p.mode + "ing_data_{}_{}.csv".format(self.variance_limit, self.seed))

        print("Testing_data: ")
        end = "\t"
        for i, env in enumerate(self.testing_envs.envs):
            print(env.name, testing_data[i], end=end)
            self.testing_data[env.name].append(testing_data[i])
            end = "\t" if end=="\n" else "\n"
        print(end="\n\n")

        frame = pd.DataFrame(self.testing_data)
        if self.variance_limit == 0.25:
            frame.to_csv("Generated_data/" + p.mode + "ing_test_data_{}.csv".format(self.seed))
        else:
            frame.to_csv("Generated_data/" + p.mode + "ing_test_data_{}_{}.csv".format(self.variance_limit, self.seed))

    def normalize(self, obs, g):
        obs = obs.reshape(obs.shape[0], obs.shape[1]*obs.shape[2])
        obs_norm = self.buffer.o_norm.normalize(obs)
        g_norm = self.buffer.g_norm.normalize(g)
        obs_norm = torch.FloatTensor(obs_norm).to(self.device)
        g_norm = torch.FloatTensor(g_norm).to(self.device)
        # concatenate the stuffs
        return obs_norm, g_norm
    
    # this function will choose action for the agent and do the exploration
    def add_noise(self, pi):
        action = pi.cpu().numpy().squeeze()
        # add the gaussian
        action += p.noise_eps * np.random.randn(*action.shape)
        action = np.clip(action, -1.0, 1.0)
        # random actions...
        random_actions = np.random.uniform(low = -1.0, high = 1.0, size=p.max_dof)
        # choose if use the random actions
        action += np.random.binomial(1, p.random_eps, 1)[0] * (random_actions - action)
        return action

    # soft update
    def soft_update_target_network(self):
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_((1 - p.polyak) * param.data + p.polyak * target_param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_((1 - p.polyak) * param.data + p.polyak * target_param.data)

    # update the network
    def update_network(self):
        # sample the episodes
        transitions = self.buffer.sample()
        # pre-process the observation and goal
        o, o_next, g = transitions['obs'], transitions['obs_next'], transitions['g']
        transitions['obs'], transitions['g'] = self.buffer.preproc_og(o, g)
        transitions['obs_next'], transitions['g_next'] = self.buffer.preproc_og(o_next, g)
        seq = transitions['seq']

        # start to do the update
        obs_norm, g_norm = self.normalize(transitions['obs'], transitions['g'])
        obs_next_norm, g_next_norm = self.normalize(transitions['obs_next'], transitions['g_next'])
        
        actions_tensor = torch.FloatTensor(transitions['actions']).to(self.device)
        r_tensor = torch.FloatTensor(transitions['r']).to(self.device)
           
        with torch.no_grad():
            # do the normalization
            # concatenate the stuffs
            r_tensor = r_tensor.view(p.batch_size)
            actions_next = self.actor_target(obs_next_norm, g_next_norm)
            q_next_value = self.critic_target(obs_next_norm, g_next_norm, actions_next)
            q_next_value = q_next_value.detach()
            target_q_value = r_tensor + p.gamma * q_next_value
            target_q_value = target_q_value.detach()
            # clip the q value
            clip_return = 1 / (1 - p.gamma)
            target_q_value = torch.clamp(target_q_value, -clip_return, 0)
        # the q loss
        real_q_value = self.critic(obs_norm, g_norm, actions_tensor)
        critic_loss = (target_q_value - real_q_value).pow(2).mean()
        # the actor loss
        actions_real = self.actor(obs_norm, g_norm)
        actor_loss = -self.critic(obs_norm, g_norm, actions_real).mean()
        self.a1 = actor_loss
        self.a2 = (actions_real).pow(2).mean()
        self.actions_real = actions_real  
        actor_loss += (actions_real).pow(2).mean()
        
        # start to update the network
        self.actor_optim.zero_grad()
        actor_loss.backward()

        # update the critic_network
        self.critic_optim.zero_grad()
        critic_loss.backward()
        
        self.actor_optim.step()
        self.critic_optim.step()

        return critic_loss.item(), actor_loss.item()

    # do the evaluation
    def eval_agent(self):
        training_success_rate = np.array([0.0] * len(self.envs.envs))
        
        for _ in range(p.testing_eps):
            successes = np.array([0.0]*len(self.envs.envs))
            observation = self.envs.reset()
            obs = observation['observation']
            g = observation['desired_goal']
            
            for _ in range(p.max_episode_steps):
                with torch.no_grad():
                    obs_norm, g_norm = self.normalize(obs, g)
                    pi = self.actor(obs_norm, g_norm)
                    actions = pi.detach().cpu().numpy().squeeze()
                observation_new, info = self.envs.step(actions)
                obs = observation_new['observation']
                g = observation_new['desired_goal']
                successes = successes + info['is_success']

            successes = np.array([1.0 if i else 0.0 for i in successes])
            training_success_rate = training_success_rate + successes
        training_success_rate = training_success_rate/p.testing_eps
        
        testing_success_rate = np.array([0.0] * len(self.testing_envs.envs))    
        for _ in range(p.testing_eps):
            successes = np.array([0.0]*len(self.testing_envs.envs))
            observation = self.testing_envs.reset()
            obs = observation['observation']
            g = observation['desired_goal']
            
            for _ in range(p.max_episode_steps):
                with torch.no_grad():
                    obs_norm, g_norm = self.normalize(obs, g)
                    pi = self.actor(obs_norm, g_norm)
                    actions = pi.detach().cpu().numpy().squeeze()
                observation_new, info = self.testing_envs.step(actions)
                obs = observation_new['observation']
                g = observation_new['desired_goal']
                successes = successes + info['is_success']

            successes = np.array([1.0 if i else 0.0 for i in successes])
            testing_success_rate = testing_success_rate + successes
        testing_success_rate = testing_success_rate/p.testing_eps

        return training_success_rate, testing_success_rate

