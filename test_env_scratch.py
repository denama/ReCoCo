from rtc_env import GymEnv
from stable_baselines3 import PPO, TD3, SAC
from collections import defaultdict
import pickle

import os


traces = [
          "./traces/WIRED_900kbps.json",
          "./traces/WIRED_200kbps.json",
          "./traces/WIRED_35mbps.json",
          "./traces/4G_700kbps.json",
          "./traces/4G_3mbps.json",
          "./traces/4G_500kbps.json",
          "./traces/5G_12mbps.json",
          "./traces/5G_13mbps.json",
          "./traces/trace_300k.json",
           ]


model_folder = os.path.join("./data", "model_train_bla.zip")

delay_states = False


step_time = 200
reward_profile = 0
normalize_states = True

alg_name = "SAC"
suffix = "bla"


rates_delay_loss = {}


for i in range(1):

    print(f"Trace {i+1}/{len(traces)}...")
    trace = traces[i]
    print("Testing on trace: ", trace)
    rates_delay_loss[trace] = defaultdict(list)


    env = GymEnv(step_time=step_time, input_trace=trace, normalize_states=normalize_states, reward_profile=reward_profile,
                 delay_states=delay_states)

    if alg_name == "PPO":
        model = PPO.load(model_folder, env=env)
    elif alg_name == "SAC":
        model = SAC.load(model_folder, env=env)
    elif alg_name == "TD3":
        model = TD3.load(model_folder, env=env)

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
        rates_delay_loss[trace]["Ru"].append(env.Ru)
        rates_delay_loss[trace]["Rd"].append(env.Rd)
        rates_delay_loss[trace]["Rl"].append(env.Rl)
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

    with open(f"./output/rates_delay_loss_test_{suffix}.pickle", "wb") as f:
        pickle.dump(rates_delay_loss, f)
