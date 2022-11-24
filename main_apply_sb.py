
import gym
import numpy as np
import os
from stable_baselines3 import A2C, SAC, PPO, TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from rtc_env_sb import GymEnv


#USE SAVED MODEL AND EVALUATE WITH evaluate_policy()
save_dir = "./data/temp"
os.makedirs(save_dir, exist_ok=True)

#Use saved model
loaded_model = PPO.load(save_dir + "/PPO_model")
env = GymEnv()
# env = make_vec_env(lambda: env, n_envs=1)
# Check that the prediction is the same after loading (for the same observation)
mean_reward, std_reward = evaluate_policy(loaded_model, env, n_eval_episodes=10)
print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

