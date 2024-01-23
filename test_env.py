import os
import logging
from collections import defaultdict
import pickle

from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from rtc_env import GymEnv
from stable_baselines3 import PPO, A2C, TD3, SAC

from conf_dict_util import conf_to_dict, dict_to_conf
from best_algs import best_models_dict, one_conf_models_dict

from trace_lists import low_bandwidth_traces, medium_bandwidth_traces, high_bandwidth_traces, opennetlab_traces, train, test, all_traces

def parse_trace_name_from_path(trace_path):
    return trace_path.split("/")[-1].split(".")[0]

def parse_dataset_from_path(trace_path):
    if "ghent" in trace_path.lower():
        return "ghent"
    elif "NY" in trace_path:
        return "NY"
    elif "norway" in trace_path.lower():
        return "norway"
    elif "traces" in trace_path:
        return "opennetlab"


#---------------------------------------------------------------

traces = all_traces
# model_num = 100 * 10000
model_num = 5200000
model_folder = f"./data/random_train_80_20_v2/{model_num}"
suffix = "random_train_80_20_v2_ep65"

#---------------------------------------------------------------

# print("Testing for all these traces:\n", traces)
print(f"Testing model num {model_num} from {model_folder}, suffix: {suffix}, Len traces to test {len(traces)}")


n_steps = 78280

delay_states = True
normalize_states = True
step_time = 200
alg_name = "TD3"
tuned = False
reward_profile = 0


rates_delay_loss = {}


for i in range(len(traces)):
        
    trace = traces[i]
    
    #If GCC results for that trace do not exist, do not test for it
    trace_name = parse_trace_name_from_path(trace)
    dataset = parse_dataset_from_path(trace)
    pickle_path_gcc = f"./apply_model/results_gcc/{dataset}/rates_delay_loss_gcc_{trace_name}.pickle"
    if not os.path.exists(pickle_path_gcc):
        print(f"GCC run doesn't exist or path wrong for trace {trace_name}, continuing..")
        continue
        

    print(f"Trace {i+1}/{len(traces)}...")
    print("Testing on trace: ", trace)
    rates_delay_loss[trace] = defaultdict(list)

    env = GymEnv(step_time=step_time, input_trace=trace, normalize_states=normalize_states,
                 reward_profile=reward_profile, delay_states=delay_states)

    if alg_name == "PPO":
        model = PPO.load(model_folder, env=env)
    elif alg_name == "SAC":
        model = SAC.load(model_folder, env=env)
    elif alg_name == "TD3":
        model = TD3.load(model_folder, env=env)

    obs = env.reset()
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
        # rates_delay_loss[trace]["Ru"].append(env.Ru)
        # rates_delay_loss[trace]["Rd"].append(env.Rd)
        # rates_delay_loss[trace]["Rl"].append(env.Rl)
        cumulative_reward += reward

        if done:
            print("Step ", step)
            # Note that the VecEnv resets automatically
            # when a done signal is encountered
            print("Len rates_delay_loss", len(rates_delay_loss[trace]["bandwidth_prediction"]))
            # print("Len rates_delay_loss", len(rates_delay_loss[trace]["sending_rate"]))
            print("Goal reached!",
                  "current reward=",round(reward, 3),
                  "avg reward=",round(cumulative_reward/step, 3),
                  "cumulative reward=",round(cumulative_reward, 3),
                  "trace= ", trace)
            obs = env.reset()
            break

    with open(f"./output/rates_delay_loss_test_{suffix}.pickle", "wb") as f:
        pickle.dump(rates_delay_loss, f)
