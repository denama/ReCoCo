from stable_baselines3.common.env_checker import check_env

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from rtc_env import GymEnv
from stable_baselines3 import PPO, A2C, TD3, SAC
import logging
from collections import defaultdict
import pickle
from conf_dict_util import conf_to_dict, dict_to_conf
from best_algs import best_models_dict, one_conf_models_dict


import os

#Original traces
# traces = [
#           "./traces/WIRED_900kbps.json",
#           "./traces/WIRED_200kbps.json",
#           "./traces/WIRED_35mbps.json",
#           "./traces/4G_700kbps.json",
#           "./traces/4G_3mbps.json",
#           "./traces/4G_500kbps.json",
#           "./traces/5G_12mbps.json",
#           "./traces/5G_13mbps.json",
#           "./traces/trace_300k.json",
#            ]

# #low bandwidth
# traces = \
# ['./traces/trace_300k.json',
#  './traces/4G_500kbps.json',
#  './traces/WIRED_200kbps.json',
#  './new_data/Norway_3G_data_json/tram_2010-12-22_0800CET.json',
#  './new_data/Norway_3G_data_json/tram_2011-01-04_0820CET.json',
#  './new_data/Norway_3G_data_json/metro_2010-09-14_1038CEST.json',
#  './new_data/Norway_3G_data_json/ferry_2010-09-21_1622CEST.json',
#  './new_data/Norway_3G_data_json/metro_2010-10-18_0951CEST.json',
#  './new_data/Norway_3G_data_json/train_2011-02-14_1728CET.json',
#  './new_data/Norway_3G_data_json/metro_2010-09-14_2303CEST.json',
#  './new_data/Norway_3G_data_json/tram_2010-12-22_0849CET.json',
#  './new_data/Norway_3G_data_json/bus_2010-11-10_1726CET.json',
#  './new_data/Norway_3G_data_json/bus_2011-01-31_1025CET.json',
#  './new_data/Norway_3G_data_json/bus_2010-11-10_1424CET.json',
#  './new_data/Norway_3G_data_json/tram_2010-12-09_1334CET.json',
#  './new_data/Norway_3G_data_json/tram_2010-12-21_1225CET.json',
#  './new_data/Norway_3G_data_json/bus_2010-09-30_1114CEST.json',
#  './new_data/Norway_3G_data_json/tram_2011-02-02_1345CET.json',
#  './new_data/Norway_3G_data_json/car_2011-02-14_2051CET.json',
#  './new_data/Norway_3G_data_json/bus_2010-09-30_1113CEST.json',
#  './new_data/Norway_3G_data_json/tram_2010-12-21_1134CET.json',
#  './new_data/Norway_3G_data_json/train_2011-02-11_1530CET.json',
#  './new_data/Norway_3G_data_json/bus_2011-01-30_1323CET.json',
#  './new_data/Norway_3G_data_json/tram_2010-12-16_1100CET.json',
#  './new_data/Norway_3G_data_json/ferry_2010-09-29_0702CEST.json',
#  './new_data/Norway_3G_data_json/metro_2010-09-13_1046CEST.json',
#  './new_data/Norway_3G_data_json/metro_2011-02-01_1800CET.json',
#  './new_data/Norway_3G_data_json/metro_2011-02-02_1251CET.json',
#  './new_data/Norway_3G_data_json/bus_2010-09-29_1622CEST.json',
#  './new_data/Norway_3G_data_json/car_2011-04-21_1135CEST.json',
#  './new_data/Norway_3G_data_json/ferry_2010-09-28_1003CEST.json',
#  './new_data/Norway_3G_data_json/tram_2010-12-16_1215CET.json',
#  './new_data/Norway_3G_data_json/train_2011-02-11_1618CET.json',
#  './new_data/Norway_3G_data_json/ferry_2011-02-01_0740CET.json',
#  './new_data/Norway_3G_data_json/car_2011-02-14_2139CET.json',
#  './new_data/Norway_3G_data_json/tram_2011-01-06_0814CET.json',
#  './new_data/Norway_3G_data_json/tram_2011-01-05_0819CET.json',
#  './new_data/Norway_3G_data_json/tram_2010-12-22_0826CET.json',
#  './new_data/Norway_3G_data_json/ferry_2011-01-31_1830CET.json',
#  './new_data/Norway_3G_data_json/metro_2010-11-11_1012CET.json',
#  './new_data/Norway_3G_data_json/ferry_2011-02-01_1639CET.json',
#  './new_data/Norway_3G_data_json/tram_2011-01-06_0749CET.json',
#  './new_data/Norway_3G_data_json/bus_2010-09-29_1628CEST.json',
#  './new_data/Norway_3G_data_json/ferry_2010-09-23_1001CEST.json',
#  './new_data/Norway_3G_data_json/ferry_2010-09-21_1735CEST.json',
#  './new_data/Norway_3G_data_json/metro_2011-01-31_2356CET.json',
#  './new_data/Norway_3G_data_json/bus_2010-09-30_1058CEST.json',
#  './new_data/Norway_3G_data_json/metro_2010-09-27_0942CEST.json',
#  './new_data/Norway_3G_data_json/metro_2010-09-22_0857CEST.json',
#  './new_data/Norway_3G_data_json/metro_2010-09-13_1003CEST.json',
#  './new_data/Norway_3G_data_json/tram_2010-12-09_1222CET.json',
#  './new_data/Norway_3G_data_json/bus_2011-01-29_1423CET.json',
#  './new_data/Norway_3G_data_json/ferry_2011-02-01_0629CET.json',
#  './new_data/Norway_3G_data_json/metro_2010-11-16_1857CET.json',
#  './new_data/Norway_3G_data_json/bus_2011-01-29_1125CET.json',
#  './new_data/Norway_3G_data_json/train_2011-02-14_0644CET.json',
#  './new_data/Norway_3G_data_json/ferry_2010-09-22_0702CEST.json',
#  './new_data/Norway_3G_data_json/train_2011-02-11_1729CET.json',
#  './new_data/Norway_3G_data_json/metro_2010-11-04_0957CET.json',
#  './new_data/Norway_3G_data_json/metro_2011-01-31_1935CET.json',
#  './new_data/Norway_3G_data_json/bus_2010-09-30_1133CEST.json',
#  './new_data/Norway_3G_data_json/bus_2010-09-29_1823CEST.json',
#  './new_data/Norway_3G_data_json/ferry_2010-09-21_1001CEST.json',
#  './new_data/Norway_3G_data_json/bus_2011-01-29_1827CET.json',
#  './new_data/Norway_3G_data_json/tram_2010-12-21_1200CET.json',
#  './new_data/Norway_3G_data_json/tram_2010-12-09_1310CET.json',
#  './new_data/Norway_3G_data_json/car_2011-02-14_2108CET.json',
#  './new_data/Norway_3G_data_json/tram_2010-12-16_1125CET.json',
#  './new_data/Norway_3G_data_json/ferry_2011-02-01_1000CET.json',
#  './new_data/Norway_3G_data_json/ferry_2010-09-20_1542CEST.json',
#  './new_data/Norway_3G_data_json/ferry_2011-02-01_0840CET.json',
#  './new_data/Norway_3G_data_json/car_2011-02-14_2032CET.json',
#  './new_data/Norway_3G_data_json/tram_2010-12-09_1244CET.json',
#  './new_data/Norway_3G_data_json/tram_2010-11-23_1515CET.json',
#  './new_data/Norway_3G_data_json/tram_2010-11-23_1606CET.json',
#  './new_data/Norway_3G_data_json/tram_2011-01-31_2032CET.json',
#  './new_data/Norway_3G_data_json/tram_2010-11-23_1541CET.json',
#  './new_data/Norway_3G_data_json/ferry_2011-01-29_1800CET.json',
#  './new_data/Norway_3G_data_json/tram_2010-12-16_1149CET.json',
#  './new_data/Norway_3G_data_json/metro_2010-10-22_1458CEST.json',
#  './new_data/Norway_3G_data_json/tram_2011-01-31_1045CET.json',
#  './new_data/Norway_3G_data_json/metro_2010-09-21_0742CEST.json',
#  './new_data/Norway_3G_data_json/ferry_2011-02-01_1539CET.json',
#  './new_data/Norway_3G_data_json/car_2011-02-10_1611CET.json',
#  './new_data/Norway_3G_data_json/car_2011-02-14_2124CET.json',
#  './new_data/Norway_3G_data_json/bus_2010-09-29_1827CEST.json',
#  './new_data/Norway_3G_data_json/bus_2010-09-29_0852CEST.json',
#  './new_data/Norway_3G_data_json/metro_2010-09-14_1415CEST.json',
#  './new_data/Norway_3G_data_json/bus_2010-09-28_1407CEST.json',
#  './new_data/NY_4G_data_json/Car_Car_1.json']

#medium bandwidth
traces = \
['./traces/4G_3mbps.json',
 './traces/WIRED_900kbps.json',
 './traces/4G_700kbps.json',
 './new_data/logs_all_4G_Ghent_json/report_bus_0004.json',
 './new_data/logs_all_4G_Ghent_json/report_car_0001.json',
 './new_data/logs_all_4G_Ghent_json/report_car_0007.json',
 './new_data/logs_all_4G_Ghent_json/report_car_0006.json',
 './new_data/logs_all_4G_Ghent_json/report_foot_0002.json',
 './new_data/logs_all_4G_Ghent_json/report_foot_0005.json',
 './new_data/logs_all_4G_Ghent_json/report_tram_0007.json',
 './new_data/logs_all_4G_Ghent_json/report_train_0003.json',
 './new_data/logs_all_4G_Ghent_json/report_bicycle_0001.json',
 './new_data/logs_all_4G_Ghent_json/report_tram_0003.json',
 './new_data/logs_all_4G_Ghent_json/report_bus_0001.json',
 './new_data/logs_all_4G_Ghent_json/report_tram_0006.json',
 './new_data/logs_all_4G_Ghent_json/report_car_0008.json',
 './new_data/logs_all_4G_Ghent_json/report_bus_0010.json',
 './new_data/logs_all_4G_Ghent_json/report_tram_0005.json',
 './new_data/logs_all_4G_Ghent_json/report_foot_0008.json',
 './new_data/logs_all_4G_Ghent_json/report_tram_0004.json',
 './new_data/logs_all_4G_Ghent_json/report_foot_0001.json',
 './new_data/logs_all_4G_Ghent_json/report_car_0004.json',
 './new_data/logs_all_4G_Ghent_json/report_foot_0003.json',
 './new_data/logs_all_4G_Ghent_json/report_foot_0007.json',
 './new_data/logs_all_4G_Ghent_json/report_car_0002.json',
 './new_data/logs_all_4G_Ghent_json/report_bus_0008.json',
 './new_data/logs_all_4G_Ghent_json/report_car_0003.json',
 './new_data/logs_all_4G_Ghent_json/report_tram_0008.json',
 './new_data/logs_all_4G_Ghent_json/report_bus_0006.json',
 './new_data/logs_all_4G_Ghent_json/report_bicycle_0002.json',
 './new_data/logs_all_4G_Ghent_json/report_bus_0011.json',
 './new_data/logs_all_4G_Ghent_json/report_bus_0005.json',
 './new_data/logs_all_4G_Ghent_json/report_bus_0003.json',
 './new_data/logs_all_4G_Ghent_json/report_car_0005.json',
 './new_data/logs_all_4G_Ghent_json/report_tram_0002.json',
 './new_data/logs_all_4G_Ghent_json/report_train_0002.json',
 './new_data/logs_all_4G_Ghent_json/report_foot_0006.json',
 './new_data/logs_all_4G_Ghent_json/report_bus_0002.json',
 './new_data/logs_all_4G_Ghent_json/report_bus_0007.json',
 './new_data/logs_all_4G_Ghent_json/report_train_0001.json',
 './new_data/logs_all_4G_Ghent_json/report_foot_0004.json',
 './new_data/logs_all_4G_Ghent_json/report_tram_0001.json',
 './new_data/NY_4G_data_json/Bus_B57_bus57_1.json',
 './new_data/NY_4G_data_json/Ferry_Ferry2.json',
 './new_data/NY_4G_data_json/Subway_7Train_7Train1.json',
 './new_data/NY_4G_data_json/LIRR_Long_Island_Rail_Road.json',
 './new_data/NY_4G_data_json/Ferry_Ferry3.json',
 './new_data/NY_4G_data_json/BusBrooklyn_bus62New.json',
 './new_data/NY_4G_data_json/7Train_7trainNew.json',
 './new_data/NY_4G_data_json/Ferry_Ferry4.json',
 './new_data/NY_4G_data_json/Bus_B62_bus62.json',
 './new_data/NY_4G_data_json/Bus_M15_M15_1.json',
 './new_data/NY_4G_data_json/QTrain_QtrainNew.json',
 './new_data/NY_4G_data_json/Bus_B62_bus62_2.json',
 './new_data/NY_4G_data_json/Ferry_Ferry1.json',
 './new_data/NY_4G_data_json/Ferry_Ferry5.json']

#high bandwidth
# traces = \
# ['./traces/5G_13mbps.json',
#  './traces/5G_12mbps.json',
#  './traces/WIRED_35mbps.json',
#  './new_data/logs_all_4G_Ghent_json/report_bus_0009.json',
#  './new_data/NY_4G_data_json/7Train_7BtrainNew.json',
#  './new_data/NY_4G_data_json/Subway_D_Train_d1.json',
#  './new_data/NY_4G_data_json/Bus_NYU_Campus_NYU_Campus_Bus.json',
#  './new_data/NY_4G_data_json/Bus_M15_M15_2.json',
#  './new_data/NY_4G_data_json/Subway_Q_Train_Q_Train3.json',
#  './new_data/NY_4G_data_json/Bus_B57_bus57_2.json',
#  './new_data/NY_4G_data_json/Subway_D_Train_d2.json',
#  './new_data/NY_4G_data_json/Subway_Q_Train_Q_Train2.json',
#  './new_data/NY_4G_data_json/Subway_7Train_7Train2.json',
#  './new_data/NY_4G_data_json/Car_Car_2.json',
#  './new_data/NY_4G_data_json/BusBrooklyn_bus57New.json',
#  './new_data/NY_4G_data_json/Subway_Q_Train_Q_Train1.json']


#Ghent 4G
# base_path = "./new_data/logs_all_4G_Ghent_json"

#Norway 3G
# base_path = "./new_data/Norway_3G_data_json"

#NY 4G
# base_path = "./new_data/NY_4G_data_json"

# traces = [os.path.join(base_path, file) for file in os.listdir(base_path) \
#          if "json" in file]


print("Testing for all these traces:\n", traces)

model_num = 1000000
model_folder = f"./data/random_medium_bandwidth/{model_num}.zip"
suffix = "random_medium_bandwidth"

# n_steps = 61118 #Norway
n_steps = 78280 #NY

delay_states = True
normalize_states = True
step_time = 200
alg_name = "TD3"
tuned = False
reward_profile = 0


rates_delay_loss = {}


for i in range(len(traces)):

    print(f"Trace {i+1}/{len(traces)}...")
    trace = traces[i]
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
