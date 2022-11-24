
import gym
import numpy as np
import logging
import os

from stable_baselines3 import A2C, SAC, PPO, TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecNormalize

from rtc_env_sb import GymEnv


save_dir = "./data/temp"
os.makedirs(save_dir, exist_ok=True)

#PPO hyperparams
learning_rate = 0.001

logging.basicConfig(filename='logs/main_train_sb.log', level=logging.INFO, filemode='w')
logging.info('Started main SB')

# Random Agent, before training

print("Training PPO")
env = GymEnv()
env = make_vec_env(lambda: env, n_envs=5)
#Trial 1 with this normalization
# env = VecNormalize(env, training=True, norm_obs=True, norm_reward=True, clip_obs=1.0, clip_reward=1.0)
model_PPO = PPO("MlpPolicy", env, verbose=1, learning_rate=learning_rate, tensorboard_log="./tensorboard_logs/")
num_episodes = 2
model_PPO.learn(total_timesteps=4000 * num_episodes, progress_bar=True)
model_PPO.save(save_dir + "/PPO_model")

#Evaluate model
# mean_reward, std_reward = evaluate_policy(model_PPO, env, n_eval_episodes=10)
# print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

# print("Training A2C")
# eval_env = GymEnv()
# eval_env = make_vec_env(lambda: eval_env, n_envs=1)
# model_A2C = A2C('MlpPolicy', eval_env, verbose=1, gamma=0.9, n_steps=20)
# model_A2C.learn(total_timesteps=5000, progress_bar=True)
# model_A2C.save(save_dir + "/A2C_model")


# # sample an observation from the environment
# obs = model.env.observation_space.sample()






# def evaluate(model, num_episodes=10, deterministic=True):
#     """
#     Evaluate a RL agent
#     :param model: (BaseRLModel object) the RL Agent
#     :param num_episodes: (int) number of episodes to evaluate it
#     :return: (float) Mean reward for the last num_episodes
#     """
#     # This function will only work for a single Environment
#     env = model.get_env()
#     all_episode_rewards = []
#     avg_episode_rewards = []
#     for i in range(num_episodes):
#         episode_rewards = []
#         done = False
#         obs = env.reset()
#         while not done:
#             # _states are only useful when using LSTM policies
#             action, _states = model.predict(obs, deterministic=deterministic)
#             # here, action, rewards and dones are arrays
#             # because we are using vectorized env
#             obs, reward, done, info = env.step(action)
#             episode_rewards.append(reward)
#
#         all_episode_rewards.append(sum(episode_rewards))
#         avg_episode_rewards.append(np.mean(episode_rewards))
#
#     mean_episode_reward = np.mean(all_episode_rewards)
#     print("Mean reward:", mean_episode_reward, "Num episodes:", num_episodes)
#
#     return mean_episode_reward


# obs = env.reset()
#
# for i in range(300):
#     action, _states = model.predict(obs, deterministic=True)
#     obs, reward, done, info = env.step(action)
#     print('obs=', obs, 'reward=', reward, 'done=', done)
#     if done:
#       obs = env.reset()
#       break
#
# env.close()