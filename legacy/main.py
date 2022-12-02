#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import torch
import matplotlib.pyplot as plt

import draw
from draw import plot_avg_reward, plot_total_reward, plot_cumulative_reward
from rtc_env import GymEnv
from deep_rl.storage import Storage
from deep_rl.ppo_agent import PPO

from collections import defaultdict
import pickle
import logging

np.warnings.filterwarnings('ignore')

logging.basicConfig(filename='logs/main_train.log', level=logging.INFO, filemode='w')
logging.info('Started main')

def main():
    ############## Hyperparameters for the experiments ##############
    env_name = "AlphaRTC"
    max_num_episodes = 20     # maximal episodes
    save_interval = 2          # save model and plot stuff every save_interval episode

    update_interval = 4000      # update policy every update_interval timesteps (4000 steps per episode)
    step_size = 200             # how many ms in one step (we change rate every step)
    exploration_param = 0.05    # the std var of action distribution
    K_epochs = 37               # update policy for K_epochs
    ppo_clip = 0.2              # clip parameter of PPO
    gamma = 0.99                # discount factor

    lr = 0.001                 # Adam parameters
    betas = (0.9, 0.999)
    # state_dim = (4,5)
    state_dim = 4*5
    action_dim = 1
    data_path = f'./data/'      # Save model and reward curve here
    #############################################

    if not os.path.exists(data_path):
        os.makedirs(data_path)

    env = GymEnv(step_size)
    # print(f"ENV INFO: Action space {env.action_space}, observation space {env.observation_space}")
    storage = Storage() # used for storing data

    ppo = PPO(state_dim=state_dim, action_dim=action_dim, exploration_param=exploration_param,
              lr=lr, betas=betas, gamma=gamma,
              ppo_epoch=K_epochs, ppo_clip=ppo_clip, retrain=True)

    record_episode_reward = []
    record_avg_episode_reward = []
    episode_reward = 0
    time_step = 0

    write_out = False

    # training loop
    #episode loop
    for episode in range(max_num_episodes):
        # print("%%%%%%%%%%---------START OF AN EPISODE---------------%%%%%%%%%%%%%%%%%%%%%%")
        logging.info(f"Start of episode {episode}")
        epoch_counter = 0
        if episode % 1 == 0:
            write_out = True
            # logging.info(f"WRITE OUT {write_out}")
        else:
            write_out = False
        #epoch loop
        while time_step < update_interval:
            done = False
            #By resetting state we are starting with a new random tracefile
            #If you want to try to train on one trace only, put training=False and set trace in rtc_env
            state = env.reset(training=True)
            # state = env.state

            # print("Time step at reset: ", time_step)

            #step in epoch loop
            while not done and time_step < update_interval:
                action = ppo.select_action(state, storage)
                state, reward, done, _ = env.step(action)

                logging.info(f"Reward: {reward}")
                if (reward < -1) or (reward > 1):
                    print("Reward is not between -1 and 1!")

                # Collect data for update
                storage.rewards.append(reward)
                storage.is_terminals.append(done)
                time_step += 1
                episode_reward += reward

            env.clear_list_of_packets()
            epoch_counter += 1

        # logging.info(f"Episode num {episode} ends.")

        #This is to calculate the reward and update the policy at the end of the episode
        state = torch.FloatTensor(state)
        next_value = ppo.get_value(state)
        storage.compute_returns(next_value, gamma)
        print("%%%%%%%%%%---------END OF AN EPISODE---------------%%%%%%%%%%%%%%%%%%%%%%")
        print("Time step", time_step)

        # update policy
        policy_loss, val_loss = ppo.update(storage)
        storage.clear_storage()

        avg_episode_reward = episode_reward/time_step

        record_avg_episode_reward.append(avg_episode_reward)
        record_episode_reward.append(episode_reward)
        print(f'Episode {episode} \t Average policy loss {policy_loss}, value loss {val_loss}, '
              f'average reward per time step {avg_episode_reward}, total reward in episode {episode_reward}')

        if episode > 0 and not (episode % save_interval):
            print(f"Saving model and plotting reward ... episode {episode}")
            ppo.save_model(data_path)

            #Plot
            plot_avg_reward(record_avg_episode_reward, data_path)
            plot_total_reward(record_episode_reward, data_path)
            plot_cumulative_reward(record_episode_reward, data_path)


        episode_reward = 0
        time_step = 0

    #End code inside episode loop

    #TODO - this step is applying model and drawing
    # draw.draw_module(ppo.policy, data_path)




if __name__ == '__main__':

    main()
