import torch
import os, time
from datetime import datetime
import numpy as np
from replay_buffer import replay_buffer
from agent import Actor, Critic
import pandas as pd
from parameters import hyperparameters as p

class Trainer:
    def __init__(self, env,  seed):
        self.seed = seed
        self.successes = []
        self.epochs = []
        self.env = env
        self.device = torch.device(p.device)
        # create the network
        self.actor = Actor(self.env.ob_shape, self.env.goal_shape, self.env.action_shape).to(self.device)
        self.critic = Critic(self.env.ob_shape, self.env.goal_shape, self.env.action_shape).to(self.device)
        # build up the target network
        self.actor_target = Actor(self.env.ob_shape, self.env.goal_shape, self.env.action_shape).to(self.device)
        self.critic_target = Critic(self.env.ob_shape, self.env.goal_shape, self.env.action_shape).to(self.device)
        # load the weights into the target networks
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        # if use gpu
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=p.lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=p.lr)
        # her sampler
        self.buffer = replay_buffer(self.env.ob_shape, self.env.action_shape)
        
    def start(self):
        for self.epoch in range(p.n_epochs):
            for _ in range(p.n_cycles):
                mb_obs, mb_ag, mb_g, mb_obs_next, mb_ag_next, mb_actions = [], [], [], [], [], []
                for _ in range(1):
                    # reset the rollouts
                    ep_obs, ep_ag, ep_g, ep_obs_next, ep_ag_next, ep_actions = [], [], [], [], [], []
                    # reset the environment
                    observation = self.env.reset()
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
                        observation_new, _, _, info = self.env.step(action)
                        obs_new = observation_new['observation']
                        ag_new = observation_new['achieved_goal']
                        # append rollouts
                        ep_obs.append(obs.copy())
                        ep_ag.append(ag.copy())
                        ep_g.append(g.copy())
                        ep_obs_next.append(obs_new.copy())
                        ep_ag_next.append(ag_new.copy())
                        ep_actions.append(action.copy())
                        # re-assign the observation
                        obs = obs_new
                        ag = ag_new

                    mb_obs.append(ep_obs)
                    mb_ag.append(ep_ag)
                    mb_g.append(ep_g)
                    mb_obs_next.append(ep_obs_next)
                    mb_ag_next.append(ep_ag_next)
                    mb_actions.append(ep_actions)


                # convert them into arrays
                mb_obs = np.array(mb_obs)
                mb_ag = np.array(mb_ag)
                mb_g = np.array(mb_g)
                mb_obs_next = np.array(mb_obs_next)
                mb_ag_next = np.array(mb_ag_next)
                mb_actions = np.array(mb_actions)
                # store the episodes
                self.buffer.store_episode([mb_obs, mb_ag, mb_g, mb_obs_next, mb_ag_next, mb_actions])
                self.buffer.update_normalizer([mb_obs, mb_ag, mb_g, mb_obs_next, mb_ag_next, mb_actions])
                for _ in range(p.update_per_episode):
                    # train the network
                    c_loss, a_loss = self.update_network()
                # soft update
                self.soft_update_target_network()
            # start to do the evaluation
            success_rate = self.eval_agent()
            print('[{}] epoch: {}, seed: {}, eval success rate is: {}'.format(self.env.name, self.epoch, self.seed, success_rate))
            self.save_csv(self.epoch, success_rate)
            if len(self.successes) >= 10:
            	if sum(self.successes[-10:]) == 10.0:
            		break

    def save_csv(self, epoch, success_rate):
        try:
            os.mkdir("Generated_data")
        except:
            pass
            
        self.epochs.append(epoch+1)
        self.successes.append(success_rate)

        di = {}
        di['epochs'] = self.epochs
        di["success_rate"] = self.successes

        frame = pd.DataFrame(di)
        frame.to_csv("Generated_data/{}_{}.csv".format(self.env.name, self.seed))


    def normalize(self, obs, g):
        print(self.env.name)
        time.sleep(10000)
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
        random_actions = np.random.uniform(low = -1.0, high = 1.0, size=self.env.action_shape)
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
       
        # start to do the update
        obs_norm, g_norm = self.normalize(transitions['obs'], transitions['g'])
        obs_next_norm, g_next_norm = self.normalize(transitions['obs_next'], transitions['g_next'])
        
        actions_tensor = torch.FloatTensor(transitions['actions']).to(self.device)
        r_tensor = torch.FloatTensor(transitions['r']).to(self.device)
           
        with torch.no_grad():
            # do the normalization
            # concatenate the stuffs
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
        total_success_rate = []
        for _ in range(p.testing_eps):
            total_success_rate.append(0.0)
            observation = self.env.reset()
            obs = observation['observation']
            g = observation['desired_goal']
            for _ in range(p.max_episode_steps):
                with torch.no_grad():
                    obs_norm, g_norm = self.normalize(obs, g)
                    pi = self.actor(obs_norm, g_norm)
                    # convert the actions
                    actions = pi.detach().cpu().numpy().squeeze()
                observation_new, _, _, info = self.env.step(actions)
                obs = observation_new['observation']
                g = observation_new['desired_goal']
                if info["is_success"]:
                    break
            total_success_rate[-1] = info['is_success']
        total_success_rate = round(np.array(total_success_rate).mean(),2)
        return total_success_rate