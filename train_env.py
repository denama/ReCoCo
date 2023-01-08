from stable_baselines3.common.env_checker import check_env

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from rtc_env import GymEnv
# from rtc_env_simple import GymEnvSimple
from stable_baselines3 import PPO, A2C, TD3, SAC
import logging
from collections import defaultdict
import pickle

import os


save_dir = "./data"

delay_states = False

step_time = 200
reward_profile = 0
normalize_states = True
alg_name = "SAC"


#Trial: train it with vec_env, run it with normal env

rates_delay_loss = {}

traces = ["./traces/WIRED_900kbps.json", "./traces/WIRED_200kbps.json"]


# for i in range(len(traces)):
for i in range(1):
    
    trace = traces[i]
    print("Input trace: ", trace)

    env = GymEnv(step_time=step_time, input_trace=trace, normalize_states=normalize_states,
                 reward_profile=reward_profile, delay_states=delay_states)

    num_envs = 1
    env = make_vec_env(lambda: env, n_envs=num_envs, seed=42)
    
    learning_rate = 0.001
    model = SAC("MlpPolicy", env, verbose=1, learning_rate=learning_rate)
    
    obs = env.reset()
    model.learn(total_timesteps=500)
    model.save(os.path.join(save_dir, "model_train_bla"))

