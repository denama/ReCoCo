
from collections import defaultdict
import pickle
import numpy as np
import os
import time
import itertools
import multiprocessing

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO, A2C, TD3, SAC
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

from rtc_env_sb import GymEnv
from rtc_env_simple import GymEnvSimple

from conf_dict_params import config_dict_grid, input_conf, hyperparams_TD3, hyperparams_SAC, hyperparams_PPO

import warnings
warnings.filterwarnings("ignore")


#Train it with vec_env for num_episodes*num_timesteps
#At the end of every episode, test it with normal env

def main_func(conf_dict):
    
    # print(conf_dict)
    
    #Parse conf_dict
    trace = conf_dict["trace"]
    delay_states = conf_dict["delay_states"]
    normalize_states = conf_dict["normalize_states"]
    step_time = conf_dict["step_time"]
    alg_name = conf_dict["alg"]
    tuned = conf_dict["tuned"]
    
    trace_name = trace.split("/")[2].split(".")[0]
    # print("Input trace: ", trace)
    
    conf_params = f"{alg_name}_{trace_name}_{step_time}_delay_{delay_states}_norm_states_{normalize_states}_tuned_{tuned}"
    
    tensorboard_dir = os.path.join(input_conf["tensorboard_dir"], conf_params)
    save_subfolder = conf_params
    suffix = conf_params
    save_dir = input_conf["save_dir"]
    num_timesteps = input_conf["num_timesteps"]
    num_episodes = input_conf["num_episodes"]
    # print("Tensorboard_dir", tensorboard_dir)
    
    rates_delay_loss = {}
    
    
    if delay_states:
        env = GymEnv(step_time=step_time, input_trace=trace, normalize_states=normalize_states)
    else:
        env = GymEnvSimple(step_time=step_time, input_trace=trace, normalize_states=normalize_states)
    
    num_envs = 1
    env = make_vec_env(lambda: env, n_envs=num_envs, seed=42)

    save_model_dir = os.path.join(save_dir, save_subfolder)
    # print("I will save model in: ", save_model_dir)

    #Define model
    if alg_name == "PPO":
        if tuned:
            model = PPO(env=env, verbose=0, tensorboard_log=tensorboard_dir, **hyperparams_PPO)
        else:
            model = PPO(policy="MlpPolicy", env=env, verbose=0, tensorboard_log=tensorboard_dir)
    elif alg_name == "SAC":
        if tuned:
            model = SAC(env=env, verbose=0, tensorboard_log=tensorboard_dir, **hyperparams_SAC)
        else:
            model = SAC(policy="MlpPolicy", env=env, verbose=0, tensorboard_log=tensorboard_dir)
    elif alg_name == "TD3":
        n_actions = env.action_space.shape[-1]
        if tuned:
            action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=0.5 * np.ones(n_actions))
            model = TD3(env=env, action_noise=action_noise, verbose=0, tensorboard_log=tensorboard_dir, **hyperparams_TD3)
        else:
            action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
            model = TD3(policy="MlpPolicy", env=env, action_noise=action_noise, verbose=0, tensorboard_log=tensorboard_dir)


    obs = env.reset()
    rates_delay_loss[trace] = {}

    #Train an episode then test it
    for m in range(num_episodes):
        # print(f"Training on trace {trace} {m}..")
        model.learn(total_timesteps=num_timesteps, reset_num_timesteps=False, tb_log_name=f"{alg_name}")
        save_model_dir_per_num_timesteps = os.path.join(save_model_dir, str((m+1)*num_timesteps))
        model.save(save_model_dir_per_num_timesteps)

        # print(f"Testing on trace... {trace}")
        rates_delay_loss[trace][m] = defaultdict(list)
        
        if delay_states:
            env_test = GymEnv(step_time=step_time, input_trace=trace, normalize_states=normalize_states)
        else:
            env_test = GymEnvSimple(step_time=step_time, input_trace=trace, normalize_states=normalize_states)

        # print(f"Testing model from {save_model_dir_per_num_timesteps}")

        if alg_name == "PPO":
            model_test = PPO.load(save_model_dir_per_num_timesteps, env=env_test)
        elif alg_name == "SAC":
            model_test = SAC.load(save_model_dir_per_num_timesteps, env=env_test)
        elif alg_name == "TD3":
            model_test = TD3.load(save_model_dir_per_num_timesteps, env=env_test)

        obs = env_test.reset()
        n_steps=2000
        cumulative_reward = 0

        for step in range(n_steps):
            action, _ = model_test.predict(obs, deterministic=True)
            obs, reward, done, info = env_test.step(action)

            rates_delay_loss[trace][m]["bandwidth_prediction"].append(env_test.bandwidth_prediction_class_var)
            rates_delay_loss[trace][m]["sending_rate"].append(env_test.sending_rate)
            rates_delay_loss[trace][m]["receiving_rate"].append(env_test.receiving_rate)
            rates_delay_loss[trace][m]["delay"].append(env_test.delay)
            rates_delay_loss[trace][m]["loss_ratio"].append(env_test.loss_ratio)
            rates_delay_loss[trace][m]["log_prediction"].append(float(env_test.log_prediction))
            rates_delay_loss[trace][m]["reward"].append(reward)
            cumulative_reward += reward

            if done:
                # Note that the VecEnv resets automatically
                # when a done signal is encountered
                # print("Step ", step)
                # print("Len rates_delay_loss", len(rates_delay_loss[trace][m]["bandwidth_prediction"]),
                #      len(rates_delay_loss[trace][m]["sending_rate"]))
                # print("Goal reached!",
                #       "current reward=",round(reward, 3),
                #       "avg reward=",round(cumulative_reward/(step+1), 3),
                #       "cumulative reward=",round(cumulative_reward, 3),
                #       "trace=", trace)
                # obs = env.reset()
                break

        with open(os.path.join(input_conf["rates_delay_loss_dir"], f"rates_delay_loss_{suffix}.pickle"), "wb") as f:
            pickle.dump(rates_delay_loss, f)
            
        print("Finised with: ", conf_dict)
            
            

if __name__ == "__main__":
    
    keys, values = zip(*config_dict_grid.items())
    permutation_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
    print("Len permutation dicts: ", len(permutation_dicts))

    n_cores = input_conf["n_cores"]
    pool = multiprocessing.Pool(processes=n_cores)
    result_dict = pool.map(main_func, permutation_dicts)