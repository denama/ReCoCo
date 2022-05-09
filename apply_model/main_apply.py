#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import torch
import matplotlib.pyplot as plt

import draw
from rtc_env import GymEnv
from deep_rl.storage import Storage
from deep_rl.ppo_agent import PPO

from collections import defaultdict
import pickle
import logging

logging.basicConfig(filename='logs/model_apply.log', level=logging.INFO, filemode='w')
logging.info('Started')


############## Hyperparameters for the experiments ##############
env_name = "AlphaRTC"
max_num_episodes = 3      # maximal episodes

update_interval = 4000      # update policy every update_interval timesteps (4000 steps per episode)
save_interval = 1           # save model every save_interval episode
step_size = 100             # how many ms in one step (we change rate every step)
# exploration_param = 0.05    # the std var of action distribution
# K_epochs = 37               # update policy for K_epochs
# ppo_clip = 0.2              # clip parameter of PPO
# gamma = 0.99                # discount factor
#
# lr = 3e-5                 # Adam parameters
# betas = (0.9, 0.999)
# state_dim = 4
# action_dim = 1
data_path = f'./data/' # Save model and reward curve here
#############################################

if not os.path.exists(data_path):
    os.makedirs(data_path)

env = GymEnv(step_size)
# print(f"ENV INFO: Action space {env.action_space}, observation space {env.observation_space}")
storage = Storage() # used for storing data
# ppo = PPO(state_dim, action_dim, exploration_param, lr, betas, gamma, K_epochs, ppo_clip)

record_episode_reward = []
episode_reward = 0
time_step = 0

#Dict: episode number: list of receiving rates
record_receiving_rate_per_episode = defaultdict(dict)
record_list_of_received_packets = defaultdict(dict)

write_out = False

# training loop
for episode in range(max_num_episodes):
    # print("%%%%%%%%%%---------START OF AN EPISODE---------------%%%%%%%%%%%%%%%%%%%%%%")
    logging.info(f"Start of episode {episode}")
    epoch_counter = 0
    if episode % 4 == 0:
        write_out = True
        logging.info(f"WRITE OUT {write_out}")
    else:
        write_out = False
    while time_step < update_interval:
        done = False
        #By resetting state we are starting with a new random tracefile, every 1000 time steps ~ 2000 packets
        state = torch.Tensor(env.reset())
        # logging.info("RESETTING STATE")
        record_receiving_rate_per_episode[episode][epoch_counter] = []

        while not done and time_step < update_interval:
            action = ppo.select_action(state, storage)
            state, reward, done, _ = env.step(action)
            # print("State, reward, done before", state, reward, done)
            state = torch.Tensor(state)
            # Collect data for update
            storage.rewards.append(reward)
            storage.is_terminals.append(done)
            # print("State, reward, done after", state, reward, done)
            time_step += 1
            episode_reward += reward
            # print(f"Measured received rate {env.receiving_rate_class_var}")
            # logging.info(f"Write out receiving rate {episode} {epoch_counter} to type list {type(record_receiving_rate_per_episode[episode][epoch_counter])}")
            if write_out:
                record_receiving_rate_per_episode[episode][epoch_counter].append([env.receiving_rate_class_var, env.time])
            # print(f"State: Log of receiving rate {state[0]}, delay {state[1]}, loss ratio {state[2]}, log of latest bandwidth prediction {state[3]}")
            # print(f"Full vector state {state}")
            # logging.info(f'----------------------------Time step {time_step}')
            # logging.info(f"------------------------------Time step {time_step}, update interval {update_interval}, done {done}")
    #         if time_step == 10:
    #             logging.info("First break")
    #             break
    #     logging.info("Second break")
    #     break

        logging.info(f"Time step {time_step}")
        logging.info(f"Len list of packets {len(env.list_of_packets)}")
        logging.info(f"Len of receiving rate {len(record_receiving_rate_per_episode[episode][epoch_counter])}")
        logging.info(f"Write out list of packets in record_list_of_received_packets {episode} {epoch_counter}")
        if write_out:
            record_list_of_received_packets[episode][epoch_counter] = env.list_of_packets
        env.clear_list_of_packets()
        epoch_counter += 1

    logging.info(f"Episode num {episode} ends.")
    # logging.info("Third break")
    # # break

    next_value = ppo.get_value(state)
    storage.compute_returns(next_value, gamma)
    print("%%%%%%%%%%---------END OF AN EPISODE---------------%%%%%%%%%%%%%%%%%%%%%%")
    # print(f"Receiving rates in episode {episode}: {record_receiving_rate_per_episode[episode]}")



    # update
    policy_loss, val_loss = ppo.update(storage, state)
    storage.clear_storage()
    episode_reward /= time_step
    record_episode_reward.append(episode_reward)
    print('Episode {} \t Average policy loss {}, value loss {}, reward {}'.format(episode, policy_loss, val_loss, episode_reward))

    if episode > 0 and not (episode % save_interval):
        ppo.save_model(data_path)
        plt.plot(range(len(record_episode_reward)), record_episode_reward)
        plt.xticks(range(len(record_episode_reward)), range(len(record_episode_reward)))
        plt.xlabel('Episode')
        plt.ylabel('Averaged episode reward')
        # plt.savefig('%sreward_record.jpg' % (data_path))
        plt.savefig(os.path.join(data_path, "reward_record.jpg"))

    episode_reward = 0
    time_step = 0

#End code inside for of episode

with open("receiving_rate_per_episode.pickle", "wb") as f:
    pickle.dump(record_receiving_rate_per_episode, f)

with open("record_list_of_received_packets.pickle", "wb") as f:
    pickle.dump(record_list_of_received_packets, f)

#TODO - this step is redoing training, check
# draw.draw_module(ppo.policy, data_path)

