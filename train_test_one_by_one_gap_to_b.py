#Train on all traces introducing them one by one
#After training on each trace, test with the same trace, then move on to next trace

from collections import defaultdict
import pickle
import pandas as pd
import numpy as np
import os
import time
import itertools

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3 import PPO, A2C, TD3, SAC
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

from rtc_env import GymEnv
from conf_dict_params import config_dict_grid, input_conf, hyperparams_TD3, hyperparams_SAC, hyperparams_PPO
from conf_dict_util import conf_to_dict
from best_algs import one_conf_models_dict



#-------------------------
#Curriculum based on gap to baseline
traces = [
    "./traces/4G_3mbps.json",
    "./traces/trace_300k.json",
    "./traces/5G_12mbps.json",
    "./traces/5G_13mbps.json",
    "./traces/WIRED_900kbps.json",
    "./traces/WIRED_35mbps.json",
    "./traces/4G_700kbps.json",
    "./traces/WIRED_200kbps.json",
    "./traces/4G_500kbps.json",
           ]




tensorboard_dir = "./tensorboard_logs/gap_to_baseline_v4/"
save_subfolder = "gap_to_baseline_v4"
suffix = f"gap_to_baseline_v4"

seed = 4

num_timesteps = input_conf["num_timesteps"]
num_episodes = 30 #triple the training on each trace


save_dir = "./data"
rates_delay_loss = {}

list_conf_names = [d[200] for d in one_conf_models_dict.values()]
conff = conf_to_dict(list_conf_names[0])

delay_states = conff["delay_states"]
normalize_states = conff["normalize_states"]
step_time = conff["step_time"]
alg = conff["alg"]
tuned = conff["tuned"]
reward_profile = conff["reward_profile"]


print(f"Conf: delay states {delay_states}, norm states {normalize_states}, step time {step_time}, alg {alg}, tuned {tuned}, reward profile {reward_profile}, seed {seed}")


for i in range(len(traces)):
    
    print("------------")
    trace_path = traces[i]
    trace_name = trace_path.split("/")[2].split(".")[0]
    print("Input trace: ", trace_path, " trace name: ", trace_name)
    start = time.time()
    

    env = GymEnv(step_time=step_time, input_trace=trace_path, normalize_states=normalize_states, reward_profile=reward_profile, delay_states=delay_states)
    env = make_vec_env(lambda: env, n_envs=1, seed=seed)
    
    save_model_dir = os.path.join(save_dir, save_subfolder, str(i))
    print("I will save model in: ", save_model_dir)
    
    #Read model
    #if it's the first iteration create new model
    if i==0:
        if alg == "PPO":
            if tuned:
                model = PPO(env=env, verbose=0, tensorboard_log=tensorboard_dir, **hyperparams_PPO)
            else:
                model = PPO(policy="MlpPolicy", env=env, verbose=0, tensorboard_log=tensorboard_dir)
        elif alg == "SAC":
            if tuned:
                model = SAC(env=env, verbose=0, tensorboard_log=tensorboard_dir, **hyperparams_SAC)
            else:
                model = SAC(policy="MlpPolicy", env=env, verbose=0, tensorboard_log=tensorboard_dir)
        elif alg == "TD3":
            n_actions = env.action_space.shape[-1]
            if tuned:
                action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=0.5 * np.ones(n_actions))
                model = TD3(env=env, action_noise=action_noise, verbose=0, tensorboard_log=tensorboard_dir, **hyperparams_TD3)
            else:
                action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
                model = TD3(policy="MlpPolicy", env=env, action_noise=action_noise, verbose=0, tensorboard_log=tensorboard_dir)
    

    #read model from previous iteration
    else:
        take_model_dir = os.path.join(save_dir, save_subfolder, str(i-1))
        print("Im reading previous model from: ", take_model_dir)
        if alg == "PPO":
            model = PPO.load(take_model_dir, env=env, tensorboard_log=tensorboard_dir)
        elif alg == "SAC":
            model = SAC.load(take_model_dir, env=env, tensorboard_log=tensorboard_dir)
        elif alg == "TD3":
            model = TD3.load(take_model_dir, env=env, tensorboard_log=tensorboard_dir)

    #Train model
    obs = env.reset()
    print(f"Training model: {i}/ {len(traces)}")
    
    for m in range(num_episodes):
        print(f"Training on trace {trace_path} {m}..")
        model.learn(total_timesteps=num_timesteps, reset_num_timesteps=False, tb_log_name=f"{alg}")
    model.save(save_model_dir)
    
    end = time.time()
    print(f"Elapsed time to train on one trace: {end-start}")
    
    #Test model
    rates_delay_loss[trace_path] = defaultdict(list)

    env_test = GymEnv(step_time=step_time, input_trace=trace_path, normalize_states=normalize_states,
                              reward_profile=reward_profile, delay_states=delay_states)
    
    print(f"Testing on trace... {trace_path}, testing model from: {save_model_dir}")
    
    if alg == "PPO":
        model_test = PPO.load(save_model_dir, env=env_test)
    elif alg == "SAC":
        model_test = SAC.load(save_model_dir, env=env_test)
    elif alg == "TD3":
        model_test = TD3.load(save_model_dir, env=env_test)
        

    obs = env_test.reset()
    n_steps=2000
    cumulative_reward = 0
    
    for step in range(n_steps):
        action, _ = model_test.predict(obs, deterministic=True)
        obs, reward, done, info = env_test.step(action)
        
        rates_delay_loss[trace_path]["bandwidth_prediction"].append(env_test.bandwidth_prediction_class_var)
        rates_delay_loss[trace_path]["sending_rate"].append(env_test.sending_rate)
        rates_delay_loss[trace_path]["receiving_rate"].append(env_test.receiving_rate)
        rates_delay_loss[trace_path]["delay"].append(env_test.delay)
        rates_delay_loss[trace_path]["loss_ratio"].append(env_test.loss_ratio)
        rates_delay_loss[trace_path]["log_prediction"].append(float(env_test.log_prediction))
        rates_delay_loss[trace_path]["reward"].append(reward)
        rates_delay_loss[trace_path]["Ru"].append(env_test.Ru)
        rates_delay_loss[trace_path]["Rd"].append(env_test.Rd)
        rates_delay_loss[trace_path]["Rl"].append(env_test.Rl)
        cumulative_reward += reward

        if done:
            print("Step ", step)
            print("Len rates_delay_loss", len(rates_delay_loss[trace_path]["bandwidth_prediction"]),
                 len(rates_delay_loss[trace_path]["sending_rate"]))

            print("Goal reached!",
                  "current reward=",round(reward, 3),
                  "avg reward=",round(cumulative_reward/(step+1), 3),
                  "cumulative reward=",round(cumulative_reward, 3),
                  "trace=", trace_path)
            obs = env.reset()
            # Note that the VecEnv resets automatically
            # when a done signal is encountered
            break

    with open(f"./output/rates_delay_loss_{suffix}.pickle", "wb") as f:
        pickle.dump(rates_delay_loss, f)
