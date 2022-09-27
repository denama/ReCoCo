
#TODO For plots put example from one trace
#TODO calculate QoE performance metrics (read from 2 papers)
#TODO train an alorithm on many different traces
#TODO see on which traces to calculate performance (read from HRCC paper)

import logging
import pickle
from collections import defaultdict

from rtc_env import GymEnv
import torch

from deep_rl.storage import Storage
from deep_rl.ppo_agent import PPO

logging.basicConfig(filename='logs/main_apply.log', level=logging.INFO, filemode='w')
logging.info('Started apply model')

model_to_use = "/home/dena/Documents/Gym_RTC/gym-example/data/ppo_2022_09_24_23_57_55.pth"
step_size = 500  # how many ms in one step (we change rate every step)
exploration_param = 0.05  # the std var of action distribution
K_epochs = 37  # update policy for K_epochs
ppo_clip = 0.2  # clip parameter of PPO
gamma = 0.99  # discount factor

lr = 3e-5  # Adam parameters
betas = (0.9, 0.999)
state_dim = 4
action_dim = 1

max_num_steps = 1000
time_step = 0
ppo = PPO(state_dim=state_dim, action_dim=action_dim, exploration_param=exploration_param,
          lr=lr, betas=betas, gamma=gamma,
          ppo_epoch=K_epochs, ppo_clip=ppo_clip, retrain=False, model=model_to_use)

env = GymEnv(step_size)

storage = Storage()


rates_delay_loss = defaultdict(list)
epoch_counter = 0

while time_step < max_num_steps:
    done = False
    state = torch.Tensor(env.reset(training=False))
    print("Epoch counter: ", epoch_counter)
    rates_delay_loss[epoch_counter] = {"trace": env.current_trace,
                                       "bandwidth_prediction": [],
                                       "sending_rate": [],
                                       "receiving_rate": [],
                                       "delay": [],
                                       "loss_ratio": [],
                                       "log_prediction": [],
                                       "reward": []
                                                  }
    # print("Time step", time_step, "Done: ", done)
    while not done:
        # prediction = BWE.get_estimated_bandwidth()
        # action = BWE.action
        action = ppo.select_action(state, storage)
        state, reward, done, _ = env.step(action)
        # print(env.bandwidth_prediction_class_var)
        rates_delay_loss[epoch_counter]["bandwidth_prediction"].append(env.bandwidth_prediction_class_var)
        rates_delay_loss[epoch_counter]["sending_rate"].append(env.sending_rate)
        rates_delay_loss[epoch_counter]["receiving_rate"].append(env.receiving_rate)
        rates_delay_loss[epoch_counter]["delay"].append(env.delay)
        rates_delay_loss[epoch_counter]["loss_ratio"].append(env.loss_ratio)
        rates_delay_loss[epoch_counter]["log_prediction"].append(float(env.log_prediction))
        rates_delay_loss[epoch_counter]["reward"].append(reward)
        logging.info(f"{state[0]}, {state[1]}, {state[2]}, {float(state[3])}, {reward}")
        state = torch.Tensor(state)
        time_step += 1

    rates_delay_loss[epoch_counter]["list_of_packets"] = env.list_of_packets
    env.clear_list_of_packets()


    epoch_counter +=1


with open("rates_delay_loss.pickle", "wb") as f:
    pickle.dump(rates_delay_loss, f)


# from apply_model import BandwidthEstimator
# from apply_model import BandwidthEstimator_hrcc
# from apply_model import BandwidthEstimator_gcc

# model_path = "./apply_model/model/pretrained_model.pth"
# BWE = BandwidthEstimator.Estimator(model_path)

#Trial with no for cycle
# bandwidth_prediction1 = int(BWE.get_estimated_bandwidth())  #this step does model.forward(state)
# action = BWE.action
# # states = BWE.states
# env.reset(training=False)
# state, reward, done, _ = env.step(action)
# print(state, reward)

# BWE.report_states(states)

# model
# data_path
# max_num_steps = 1000
#
# record_reward = []
# record_state = []
# record_action = []
# episode_reward = 0
# time_step = 0
# tmp = model.random_action
# model.random_action = False
# while time_step < max_num_steps:
#     done = False
#     state = torch.Tensor(env.reset(training=False))
#     while not done:
#         action, _, _ = model.forward(state)
#         state, reward, done, _ = env.step(action)
#         state = torch.Tensor(state)
#         record_state.append(state)
#         record_reward.append(reward)
#         record_action.append(action)
#         time_step += 1
# model.random_action = True

# stats = {
#     "send_time_ms": 100,
#     "arrival_time_ms": 400,
#     "payload_type": 125,
#     "sequence_number": 10,
#     "ssrc": 123,
#     "padding_length": 0,
#     "header_length": 120,
#     "payload_size": 1350
# }
#
# # Challenge example estimator - an RF model using a trained model "pretrained_model.pth"
# print("Challenge example estimator - an RF model with a feed-forward network")
# model_path = "./apply_model/model/pretrained_model.pth"
# BWE = BandwidthEstimator.Estimator(model_path)
# print("Reality check step time: ", BWE.step_time)
# print("Prediction 1: ", int(BWE.get_estimated_bandwidth()))
# BWE.report_states(stats)
# bandwidth_prediction1 = int(BWE.get_estimated_bandwidth())
# print("Prediction 2: ", bandwidth_prediction1)
# print("----------------------------------------------")


# ## Other estimators
# # hrcc_model_path = './apply_model/model/ppo_2021_07_25_04_57_11_with500trace.pth'
# print("HRCC estimator - chooses between GCC and RF")
# hrcc_model_path = "./apply_model/model/ppo_2022_05_05_16_19_31_v2.pth"
# BWE_hrcc = BandwidthEstimator_hrcc.Estimator(hrcc_model_path)
# print("Reality check step time: ", BWE_hrcc.step_time)
# bandwidth_prediction2 = BWE_hrcc.get_estimated_bandwidth()
# print("Prediction 1: ", bandwidth_prediction2)
# BWE_hrcc.report_states(stats)
# print("Prediction 2: ", BWE_hrcc.get_estimated_bandwidth())
# print("----------------------------------------------")


# BWE_gcc = BandwidthEstimator_gcc.GCCEstimator()
# bandwidth_prediction3, _ = BWE_gcc.get_estimated_bandwidth()
# print("Reality check state: ", BWE_gcc.state)
# print("Prediction fromGCC estimator", bandwidth_prediction3)

