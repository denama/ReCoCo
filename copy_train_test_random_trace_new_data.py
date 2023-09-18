

from collections import defaultdict
import pickle
import pandas as pd
import numpy as np
import os
import time
import itertools
import multiprocessing

from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO, A2C, TD3, SAC
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

from rtc_env import GymEnv
from conf_dict_params import config_dict_grid, input_conf, hyperparams_TD3, hyperparams_SAC, hyperparams_PPO
from conf_dict_util import conf_to_dict
from best_algs import one_conf_models_dict


tensorboard_dir = f"./tensorboard_logs/random_norway_max_400_1000/"
save_subfolder = f"random_norway_max_400_1000"
suffix = f"random_norway_max_400_1000"

#If we want to train on a list, we specify the trace list
trace_set = \
['./new_data/Norway_3G_data_json/bus_2010-11-10_1424CET.json',
 './new_data/Norway_3G_data_json/bus_2010-09-30_1114CEST.json',
 './new_data/Norway_3G_data_json/bus_2010-09-29_1622CEST.json',
 './new_data/Norway_3G_data_json/ferry_2011-02-01_0740CET.json',
 './new_data/Norway_3G_data_json/ferry_2011-02-01_1639CET.json',
 './new_data/Norway_3G_data_json/ferry_2011-02-01_0629CET.json',
 './new_data/Norway_3G_data_json/bus_2010-09-29_1823CEST.json',
 './new_data/Norway_3G_data_json/ferry_2011-02-01_0840CET.json',
 './new_data/Norway_3G_data_json/bus_2010-09-29_0852CEST.json']


#If we want to train on a whole directory
take_all_from_directory = False


if take_all_from_directory:
    #Directory to take random traces from
    # trace_dir_for_random_trace = "traces"
    # trace_dir_for_random_trace = "new_data/logs_all_4G_Ghent_json"
    trace_dir_for_random_trace = "new_data/Norway_3G_data_json"
    trace_dir = os.path.join(os.path.dirname(__file__), trace_dir_for_random_trace)
    trace_set = glob.glob(f'{trace_dir}/**/*.json', recursive=True)


print("Trace set I'm working on:", trace_set)

num_timesteps = input_conf["num_timesteps"]
num_episodes = 100
continue_training = True

if not continue_training:
    start_counter = 0
else:
    model_to_continue_from = "580000.zip" #last saved model
    start_counter = 58 #num of episodes that the model has been trained on (NOT the next episode!)
seed = 1

save_dir = "./data"
rates_delay_loss = {}

# list_conf_names = [d[200] for d in one_conf_models_dict.values()]
# conff = conf_to_dict(list_conf_names[0])

# delay_states = conff["delay_states"]
# normalize_states = conff["normalize_states"]
# step_time = conff["step_time"]
# alg = conff["alg"]
# tuned = conff["tuned"]
# reward_profile = conff["reward_profile"]

delay_states = True
normalize_states = True
step_time = 200
alg = "TD3"
tuned = False
reward_profile = 0


print(f"Conf: delay states {delay_states}, norm states {normalize_states}, step time {step_time}, alg {alg}, tuned {tuned}, reward profile {reward_profile}, seed {seed}, \
        Num traces Im working with {len(trace_set)}")


rates_delay_loss = {}


start = time.time()

env = GymEnv(step_time=step_time, normalize_states=normalize_states, reward_profile=reward_profile, delay_states=delay_states,
             random_trace=True, trace_set=trace_set)
env = make_vec_env(lambda: env, n_envs=1, seed=seed)

save_model_dir = os.path.join(save_dir, save_subfolder)
print("I will save model in: ", save_model_dir)


#Define model
if continue_training:
        take_model_dir = os.path.join(save_model_dir, model_to_continue_from)
        print("Im reading previous model from: ", take_model_dir)
        if alg == "PPO":
            model = PPO.load(take_model_dir, env=env, tensorboard_log=tensorboard_dir)
        elif alg == "SAC":
            model = SAC.load(take_model_dir, env=env, tensorboard_log=tensorboard_dir)
        elif alg == "TD3":
            model = TD3.load(take_model_dir, env=env, tensorboard_log=tensorboard_dir)
else:
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
  

obs = env.reset()

# trace = "./new_data/logs_all_4G_Ghent_json/report_bicycle_0001.json"
# trace = "./new_data/Norway_3G_data_json/tram_2011-02-02_1345CET.json"
trace = "./new_data/Norway_3G_data_json/tram_2010-12-09_1334CET.json"

n_steps = 3000
rates_delay_loss[trace] = {}

print("Learning..")

#Train an episode then test it
for m in range(start_counter, start_counter+num_episodes):
    
    print("----------------")
    print(f"Training {m+1} / {start_counter+num_episodes}")
    model.learn(total_timesteps=num_timesteps, reset_num_timesteps=False, tb_log_name=f"{suffix}")
    save_model_dir_per_num_timesteps = os.path.join(save_model_dir, str((m+1)*num_timesteps))
    model.save(save_model_dir_per_num_timesteps)

    end = time.time()
    print(f"Elapsed time until now (training {num_timesteps} steps in {m+1}-th episode): {round(end-start,2)} s")
    
    
    rates_delay_loss[trace][m] = defaultdict(list)

    env_test = GymEnv(step_time=step_time, input_trace=trace, normalize_states=normalize_states, reward_profile=reward_profile,
                      delay_states=delay_states, random_trace=False)
    print(f"Testing model from {save_model_dir_per_num_timesteps} on trace {trace}..")
    
    if alg == "PPO":
        model_test = PPO.load(save_model_dir_per_num_timesteps, env=env_test)
    elif alg == "SAC":
        model_test = SAC.load(save_model_dir_per_num_timesteps, env=env_test)
    elif alg == "TD3":
        model_test = TD3.load(save_model_dir_per_num_timesteps, env=env_test)

    obs = env_test.reset()
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
            print("Testing complete!",
                  "current reward=",round(reward, 3),
                  "avg reward=",round(cumulative_reward/(step+1), 3),
                  "cumulative reward=",round(cumulative_reward, 3),
                  "trace=", trace)
            # obs = env.reset()
            break

    with open(f"./output/rates_delay_loss_{suffix}.pickle", "wb") as f:
        pickle.dump(rates_delay_loss, f)
