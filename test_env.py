from stable_baselines3.common.env_checker import check_env

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from rtc_env_sb import GymEnv
from rtc_env_simple import GymEnvSimple
from stable_baselines3 import PPO, A2C, TD3, SAC
import logging
from collections import defaultdict
import pickle

import os

#TEST FINAL MODEL HERE AND WRITE ANOTHER rates_delay_loss

save_dir = "./data"
alg_name = "SAC"




model_folder = "./data/SAC_all_traces_1_env_v6/5.zip"
suffix = "SAC_all_traces_1_env_v6"

step_time = 200

#Trial: train it with vec_env, run it with normal env

rates_delay_loss = {}

traces = ["./traces/WIRED_900kbps.json",
          "./traces/WIRED_35mbps.json",
          "./traces/WIRED_200kbps.json",
          "./traces/4G_700kbps.json",
          "./traces/4G_3mbps.json",
          "./traces/4G_500kbps.json",
           ]



for i in range(6):
    
    print("Testing..")
    
    trace = traces[i]
    print("Input trace: ", trace)
    rates_delay_loss[trace] = defaultdict(list)
    
    env = GymEnvSimple(step_time=step_time, input_trace=trace,  normalize_states=True, reward_profile=0)
    model = SAC.load(model_folder, env=env)

    obs = env.reset()
    n_steps=2000
    cumulative_reward = 0
    avg_reward = 0
    
    for step in range(n_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        
        rates_delay_loss[trace]["bandwidth_prediction"].append(env.bandwidth_prediction_class_var)
        rates_delay_loss[trace]["sending_rate"].append(env.sending_rate)
        rates_delay_loss[trace]["receiving_rate"].append(env.receiving_rate)
        rates_delay_loss[trace]["delay"].append(env.delay)
        rates_delay_loss[trace]["loss_ratio"].append(env.loss_ratio)
        rates_delay_loss[trace]["log_prediction"].append(float(env.log_prediction))
        rates_delay_loss[trace]["reward"].append(reward)
        cumulative_reward += reward

        if done:
            print("Step ", step)
            # Note that the VecEnv resets automatically
            # when a done signal is encountered
            print("Len rates_delay_loss", len(rates_delay_loss[trace]["bandwidth_prediction"]))
            print("Len rates_delay_loss", len(rates_delay_loss[trace]["sending_rate"]))
            print("Goal reached!",
                  "current reward=",round(reward, 3),
                  "avg reward=",round(cumulative_reward/step, 3),
                  "cumulative reward=",round(cumulative_reward, 3),
                  "trace= ", trace)
            obs = env.reset()
            break

    with open(f"./output/rates_delay_loss_{suffix}.pickle", "wb") as f:
        pickle.dump(rates_delay_loss, f)
