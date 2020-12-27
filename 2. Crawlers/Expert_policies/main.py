import argparse
import datetime, time
import gym, os
import numpy as np
import itertools
import torch
from sac import SAC
from replay_memory import ReplayMemory
import pandas as pd

#Loading custom envs
import Quadruped_10,Quadruped_11,Quadruped_12,Quadruped_13,Quadruped_14
import Quadruped_20,Quadruped_21,Quadruped_22,Quadruped_23,Quadruped_24,Quadruped_25,Quadruped_26
import Quadruped_31,Quadruped_32,Quadruped_33,Quadruped_34,Quadruped_35
import Quadruped_41,Quadruped_42,Quadruped_43,Quadruped_44,Quadruped_45
import Quadruped_51,Quadruped_52,Quadruped_53,Quadruped_54,Quadruped_55

import Hexapod_10,Hexapod_11,Hexapod_12,Hexapod_13,Hexapod_14, Hexapod_15, Hexapod_16
import Hexapod_20,Hexapod_21,Hexapod_22,Hexapod_23,Hexapod_24,Hexapod_25,Hexapod_26
import Hexapod_31,Hexapod_32,Hexapod_33,Hexapod_34,Hexapod_35
import Hexapod_41,Hexapod_42,Hexapod_43,Hexapod_44,Hexapod_45
import Hexapod_51,Hexapod_52,Hexapod_53,Hexapod_54,Hexapod_55

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                    help='Automaically adjust α (default: False)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=200000, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=100000, metavar='N',
                    help='size of replay buffer (default: 150000)')
parser.add_argument('--cuda', action="store_false",
                    help='run on CUDA (default: True)')
args = parser.parse_args()

# Environment
# env = NormalizedActions(gym.make(args.env_name))
env_names = ["Quadruped-v12","Quadruped-v13","Quadruped-v14",\
              "Quadruped-v20","Quadruped-v23","Quadruped-v24","Quadruped-v25",\
              "Quadruped-v31","Quadruped-v34","Quadruped-v35",\
              "Quadruped-v41","Quadruped-v42","Quadruped-v45",\
              "Quadruped-v51","Quadruped-v52","Quadruped-v53","Quadruped-v54","Quadruped-v55",\
              "Hexapod-v12","Hexapod-v13","Hexapod-v14","Hexapod-v15",\
              "Hexapod-v20","Hexapod-v23","Hexapod-v24","Hexapod-v26",\
              "Hexapod-v31","Hexapod-v34","Hexapod-v35",\
              "Hexapod-v41","Hexapod-v42","Hexapod-v45",\
              "Hexapod-v51","Hexapod-v52","Hexapod-v53","Hexapod-v54","Hexapod-v55"\
              ]

seeds = [1,2,3]

os.system("clear")

for env_name in env_names:
    for seed in seeds:
        print()
        avg_reward = 0.0

        env = gym.make(env_name)
        env.seed(seed)
        env.action_space.seed(seed)
        env._max_episode_steps = 1000

        torch.manual_seed(seed)
        np.random.seed(seed)

        # Agent
        agent = SAC(env.observation_space.shape[0], env.action_space, args)

        # Memory
        memory = ReplayMemory(args.replay_size, seed, args.cuda)
        t_start = time.time()

        # Training Loop
        total_numsteps = 0
        updates = 0

        for i_episode in itertools.count(1):
            episode_reward = 0
            episode_steps = 0
            done = False
            state = env.reset()

            while not done:
                if args.start_steps > total_numsteps:
                    action = env.action_space.sample()  # Sample random action
                else:
                    action = agent.select_action(state)  # Sample action from policy

                if len(memory) > args.batch_size:
                    # Number of updates per step in environment
                    for i in range(args.updates_per_step):
                        # Update parameters of all the networks
                        critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, args.batch_size, updates)

                        updates += 1

                next_state, reward, done, _ = env.step(action) # Step
                episode_steps += 1
                total_numsteps += 1
                episode_reward += reward

                # Ignore the "done" signal if it comes from hitting the time horizon.
                # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
                mask = 1 if episode_steps == 1000 else float(not done)

                memory.push(state, action, reward, next_state, mask) # Append transition to memory

                state = next_state

                if episode_steps == 1000:
                	done = True

            if total_numsteps > args.num_steps:
                break

            print("\033[F{}, Seed: {}, total_numsteps: {}, test_reward: {}   ".format(env_name, seed, total_numsteps, round(avg_reward, 2)))

            if i_episode % 10 == 0 and args.eval is True:
                avg_reward = 0.
                episodes = 10
                for _  in range(episodes):
                    state = env.reset()
                    episode_reward = 0
                    steps = 0
                    done = False
                    while not done:
                        steps += 1
                        action = agent.select_action(state, evaluate=True)

                        next_state, reward, done, _ = env.step(action)
                        episode_reward += reward
                        if steps == 1000:
                            done = True

                        state = next_state
                    avg_reward += episode_reward
                avg_reward /= episodes

        try:
        	frame = pd.read_csv("Generated_data/Expert_crawler.csv")
        	for col in frame.columns:
        		if col[0] == "U":
        			frame = frame.drop(col, axis=1)
        	di = {"{}_{}".format(env_name, seed): [avg_reward]}
        	f2 = pd.DataFrame(di)
        	frame = pd.concat([frame, f2],axis=1, ignore_index=False)
        	frame.to_csv('Generated_data/Expert_crawler.csv')
        except FileNotFoundError:
        	os.mkdir("Generated_data")
        	di = {"{}_{}".format(env_name, seed): [avg_reward]}
        	frame = pd.DataFrame(di)
        	frame.to_csv('Generated_data/Expert_crawler.csv')

        env.close()