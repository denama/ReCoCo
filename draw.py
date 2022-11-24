#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging

import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from rtc_env import GymEnv
from deep_rl.storage import Storage
from deep_rl.actor_critic import ActorCritic


def draw_state(record_action, record_state, path):
    plt.figure(figsize=(16,9))
    length = len(record_action)
    plt.subplot(411)
    plt.plot(range(length), record_action)
    plt.xlabel('time step')
    plt.ylabel('action')
    ylabel = ['receiving rate', 'delay', 'packet loss']
    record_state = [t.numpy() for t in record_state]
    record_state = np.array(record_state)
    logging.info(f"Record state\n{record_state}")
    for i in range(3):
        plt.subplot(411+i+1)
        plt.plot(range(length), record_state[:, i, -1])
        plt.xlabel('time step')
        plt.ylabel(ylabel[i])
    plt.tight_layout()
    plt.savefig("{}test_result.jpg".format(path))


def draw_module(model, data_path, max_num_steps = 1000):
    env = GymEnv()
    record_reward = []
    record_state = []
    record_action = []
    episode_reward = 0
    time_step = 0
    tmp = model.random_action
    model.random_action = False
    while time_step < max_num_steps:
        done = False            
        env.reset(training=True)
        state = env.state
        time_step_in_epoch = 0
        while not done:
            action, _, _ = model.forward(state.reshape(1, -1))
            state, reward, done, _ = env.step(action, time_step_in_epoch)
            record_state.append(state)
            record_reward.append(reward)
            record_action.append(action)
            time_step += 1
            time_step_in_epoch += 1
    model.random_action = True
    draw_state(record_action, record_state, data_path)


def plot_avg_reward(record_avg_episode_reward, data_path):
    plt.figure(figsize=(16, 5))
    plt.plot(range(len(record_avg_episode_reward)), record_avg_episode_reward, label="Average reward", color="indianred")
    # plt.xticks(range(len(record_avg_episode_reward)), range(len(record_avg_episode_reward)))
    plt.xlabel('Episode')
    plt.ylabel('Averaged episode reward')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(data_path, "reward_avg.jpg"))
    # plt.show()


def plot_total_reward(record_episode_reward, data_path):
    plt.figure(figsize=(16, 5))
    plt.plot(range(len(record_episode_reward)), record_episode_reward, label="Total reward", color="teal")
    # plt.xticks(range(len(record_episode_reward)), range(len(record_episode_reward)))
    plt.xlabel('Episode')
    plt.ylabel('Total episode reward')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(data_path, "reward_total.jpg"))
    # plt.show()

def plot_cumulative_reward(record_avg_episode_reward, data_path):
    plt.figure(figsize=(16, 5))
    plt.plot(range(len(record_avg_episode_reward)), np.cumsum(record_avg_episode_reward), label="Cumulative reward", color="orange")
    # plt.xticks(range(len(record_avg_episode_reward)), range(len(record_avg_episode_reward)))
    plt.xlabel('Episode')
    plt.ylabel('Cumulative episode reward')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(data_path, "reward_cumulative.jpg"))
    # plt.show()
