#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import torch
import matplotlib.pyplot as plt

import draw
from rtc_env import GymEnv
from deep_rl.storage import Storage
from deep_rl.ppo_agent import PPO


def main():
    ############## Hyperparameters for the experiments ##############
    env_name = "AlphaRTC"
    max_num_episodes = 3      # maximal episodes

    update_interval = 4000      # update policy every update_interval timesteps
    save_interval = 2          # save model every save_interval episode
    exploration_param = 0.05    # the std var of action distribution
    K_epochs = 37               # update policy for K_epochs
    ppo_clip = 0.2              # clip parameter of PPO
    gamma = 0.99                # discount factor

    lr = 3e-5                 # Adam parameters
    betas = (0.9, 0.999)
    state_dim = 4
    action_dim = 1
    data_path = f'./data/' # Save model and reward curve here
    #############################################

    if not os.path.exists(data_path):
        os.makedirs(data_path)

    env = GymEnv()
    print(f"ENV INFO: Action space {env.action_space}, observation space {env.observation_space}")
    storage = Storage() # used for storing data
    ppo = PPO(state_dim, action_dim, exploration_param, lr, betas, gamma, K_epochs, ppo_clip)

    record_episode_reward = []
    episode_reward  = 0
    time_step = 0

    # training loop
    for episode in range(max_num_episodes):
        while time_step < update_interval:
            done = False            
            state = torch.Tensor(env.reset())
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

        next_value = ppo.get_value(state)
        storage.compute_returns(next_value, gamma)

        # update
        policy_loss, val_loss = ppo.update(storage, state)
        storage.clear_storage()
        episode_reward /= time_step
        record_episode_reward.append(episode_reward)
        print('Episode {} \t Average policy loss {}, value loss {}, reward {}'.format(episode, policy_loss, val_loss, episode_reward))

        if episode > 0 and not (episode % save_interval):
            ppo.save_model(data_path)
            plt.plot(range(len(record_episode_reward)), record_episode_reward)
            plt.xlabel('Episode')
            plt.ylabel('Averaged episode reward')
            plt.savefig('%sreward_record.jpg' % (data_path))

        episode_reward = 0
        time_step = 0

    draw.draw_module(ppo.policy, data_path)


if __name__ == '__main__':
    main()
