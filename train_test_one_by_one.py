from stable_baselines3.common.env_checker import check_env

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from rtc_env import GymEnv
from stable_baselines3 import PPO, A2C, TD3, SAC
import logging
from collections import defaultdict
import pickle

import os
import time


save_dir = "./data"
alg_name = "SAC"

tensorboard_dir = "./tensorboard_logs/log_SAC_all_traces_1_env_v6/"
save_subfolder = "SAC_all_traces_1_env_v6"
suffix = f"{alg_name}_train_traces_one_by_one_v6"


#Train it with vec_env, then run it with normal env

rates_delay_loss = {}

traces = ["./traces/WIRED_900kbps.json",
          "./traces/WIRED_35mbps.json",
          "./traces/WIRED_200kbps.json",
          "./traces/4G_700kbps.json",
          "./traces/4G_3mbps.json",
          "./traces/4G_500kbps.json",
           ]


# for i in range(len(traces)):
for i in range(3,6):
    
    trace = traces[i]
    print("Input trace: ", trace)
    start = time.time()

    env = GymEnv(input_trace=trace)
    num_envs = 1
    env = make_vec_env(lambda: env, n_envs=num_envs)
    
    save_model_dir = os.path.join(save_dir, save_subfolder, str(i))
    print("I will save model in: ", save_model_dir)
    
    #Read model
    if i==0:
        learning_rate = 0.001
        model = SAC("MlpPolicy", env, verbose=1, learning_rate=learning_rate, tensorboard_log=tensorboard_dir)
    else:
        take_model_dir = os.path.join(save_dir, save_subfolder, str(i-1))
        print("Im reading previous model from: ", take_model_dir)
        model = SAC.load(take_model_dir, env=env, verbose=1, tensorboard_log=tensorboard_dir)
    
    obs = env.reset()
    print(f"Training model: {i}")
    
    #Try with tensorboard
    for m in range(10):
        print(f"Training on trace {trace} {m}..")
        model.learn(total_timesteps=10000, reset_num_timesteps=False, tb_log_name=f"{alg_name}")
    model.save(save_model_dir)
    
    end = time.time()
    print(f"Elapsed time to train on one trace: {end-start}")
    
    print(f"Testing on trace... {trace}")

    rates_delay_loss[trace] = defaultdict(list)
    
    env_test = GymEnv(input_trace=trace)
    print(f"Testing model from {save_model_dir}")
    model_test = SAC.load(save_model_dir, env=env_test)

    obs = env_test.reset()
    n_steps=2000
    for step in range(n_steps):
        action, _ = model_test.predict(obs, deterministic=True)
        obs, reward, done, info = env_test.step(action)
        
        rates_delay_loss[trace]["bandwidth_prediction"].append(env_test.bandwidth_prediction_class_var)
        rates_delay_loss[trace]["sending_rate"].append(env_test.sending_rate)
        rates_delay_loss[trace]["receiving_rate"].append(env_test.receiving_rate)
        rates_delay_loss[trace]["delay"].append(env_test.delay)
        rates_delay_loss[trace]["loss_ratio"].append(env_test.loss_ratio)
        rates_delay_loss[trace]["log_prediction"].append(float(env_test.log_prediction))
        rates_delay_loss[trace]["reward"].append(reward)

        if done:
            # Note that the VecEnv resets automatically
            # when a done signal is encountered
            print("Len rates_delay_loss", len(rates_delay_loss[trace]["bandwidth_prediction"]),
                 len(rates_delay_loss[trace]["sending_rate"]))
            print("Goal reached!", "reward=", reward, "trace=", trace)
            break

    with open(f"./output/rates_delay_loss_{suffix}.pickle", "wb") as f:
        pickle.dump(rates_delay_loss, f)
