import gym
from stable_baselines3 import A2C, PPO
import os
from rtc_env_sb import GymEnv

alg = A2C


model_save_dir = f"./data/{alg}"
tensorboard_dir = "./tensorboard_logs/"

env = GymEnv()
env.reset()

if alg == "A2C":
    model = A2C('MlpPolicy', env, verbose=1, tensorboard_log=tensorboard_dir)
elif alg == "PPO":
    learning_rate = 0.001
    model_PPO = PPO("MlpPolicy", env, verbose=1, learning_rate=learning_rate, tensorboard_log=tensorboard_dir)

TIMESTEPS = 10000
iters = 0
for i in range(30):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=alg)
    model.save(f"{model_save_dir}/{TIMESTEPS*i}")