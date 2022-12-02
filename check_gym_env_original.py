from stable_baselines3.common.env_checker import check_env

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from rtc_env_sb import GymEnv
from stable_baselines3 import PPO, A2C, TD3, SAC
import logging
from collections import defaultdict
import pickle

import os

# logging.basicConfig(filename='logs/check_gym_env.log', level=logging.INFO, filemode='w')
# logging.info('Started apply model')


# env = GymEnv()
# check_env(env, warn=True)


#1 Train an agent

# env = GymEnv()
# env = make_vec_env(lambda: env, n_envs=1)
# model = A2C('MlpPolicy', env, verbose=1).learn(5000)

# print("-----------------------")
# print("DONE WITH TRAINING")
# print("-----------------------")

#Load a saved agent
save_dir = "./data"
alg_name = "SAC"
iteration = 90000
folder_to_load = save_dir + "/" + alg_name + "/" + str(iteration)
print("Loading ", folder_to_load)

if alg_name == "PPO":
    model = PPO.load(folder_to_load)
elif alg_name == "TD3":
    model = TD3.load(folder_to_load)
elif alg_name == "SAC":
    model = SAC.load(folder_to_load)
elif alg_name == "A2C":
    model = A2C.load(folder_to_load)


# env = make_vec_env(lambda: env, n_envs=1)

# rates_delay_loss = {
#                    "bandwidth_prediction": defaultdict(list),
#                    "sending_rate": defaultdict(list),
#                    "receiving_rate": defaultdict(list),
#                    "delay": defaultdict(list),
#                    "loss_ratio": defaultdict(list),
#                    "log_prediction": defaultdict(list),
#                    "reward": defaultdict(list)
#                    }

rates_delay_loss = {}

trace1 = "./traces/WIRED_900kbps.json"
trace2 = "./traces/WIRED_35mbps.json"

env = GymEnv()

counter_trace = 0
for trace in [trace1, trace2]: 
    
    obs = env.reset(trace)

    n_steps = 1000
    
    rates_delay_loss[env.current_trace] = defaultdict(list)

    for step in range(n_steps):
        action, _ = model.predict(obs, deterministic=True)

        # print("Step {}".format(step + 1))
        # print("Action: ", action)
        obs, reward, done, info = env.step(action)
        # print('obs=', obs, 'reward=', reward, 'done=', done)
        # print("action", action)
        # rates_delay_loss["trace"][counter_trace] = env.current_trace
        # print(env.current_trace)
        rates_delay_loss[env.current_trace]["bandwidth_prediction"].append(env.bandwidth_prediction_class_var)
        rates_delay_loss[env.current_trace]["sending_rate"].append(env.sending_rate)
        rates_delay_loss[env.current_trace]["receiving_rate"].append(env.receiving_rate)
        rates_delay_loss[env.current_trace]["delay"].append(env.delay)
        rates_delay_loss[env.current_trace]["loss_ratio"].append(env.loss_ratio)
        rates_delay_loss[env.current_trace]["log_prediction"].append(float(env.log_prediction))
        rates_delay_loss[env.current_trace]["reward"].append(reward)

        if done:
            # Note that the VecEnv resets automatically
            # when a done signal is encountered
            print("Goal reached!", "reward=", reward, "counter trace=", counter_trace)
            # obs = env.reset()
            counter_trace +=1
            break

with open(f"./output/rates_delay_loss_{alg_name}_{iteration}.pickle", "wb") as f:
    pickle.dump(rates_delay_loss, f)