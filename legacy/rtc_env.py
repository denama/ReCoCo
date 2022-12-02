#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import sys
import os
import random
import numpy as np
import glob
import logging
import torch
import pandas as pd

import gym
from gym import spaces

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "gym"))
from gym_folder import alphartc_gym
from gym_folder.alphartc_gym import gym_file
from gym_folder.alphartc_gym.utils.packet_info import PacketInfo
from gym_folder.alphartc_gym.utils.packet_record import PacketRecord



UNIT_M = 1000000 #from bps to megabps
MAX_BANDWIDTH_MBPS = 8
MIN_BANDWIDTH_MBPS = 0.01
LOG_MAX_BANDWIDTH_MBPS = np.log(MAX_BANDWIDTH_MBPS)
LOG_MIN_BANDWIDTH_MBPS = np.log(MIN_BANDWIDTH_MBPS)
# print(f"MIN bandwidth log {LOG_MIN_BANDWIDTH_MBPS}, MAX bandwidth log {LOG_MAX_BANDWIDTH_MBPS}")

#Normalize value between 0 and 1 using log
def linear_to_log(value):
    # limit any value (rate) to 10kbps~8Mbps
    value = np.clip(value / UNIT_M, MIN_BANDWIDTH_MBPS, MAX_BANDWIDTH_MBPS)
    # from 10kbps~8Mbps to 0~1
    log_value = np.log(value)
    value_to_return = (log_value - LOG_MIN_BANDWIDTH_MBPS) / (LOG_MAX_BANDWIDTH_MBPS - LOG_MIN_BANDWIDTH_MBPS)
    # print(f"Rate after clip {value}, after log {log_value}, normalized between 0 and 1 {value_to_return}")
    return value_to_return


def log_to_linear(value):
    # from 0~1 to 10kbps to 8Mbps
    value = np.clip(value, 0, 1)
    log_bwe = value * (LOG_MAX_BANDWIDTH_MBPS - LOG_MIN_BANDWIDTH_MBPS) + LOG_MIN_BANDWIDTH_MBPS
    return np.exp(log_bwe) * UNIT_M


class GymEnv(gym.Env):

    def __init__(self, step_time=200):
        super(GymEnv, self).__init__()

        self.gym_env = None     
        self.step_time = step_time

        trace_dir = os.path.join(os.path.dirname(__file__), "traces")
        # trace_dir = os.path.join(os.path.dirname(__file__), "gym_folder", "alphartc_gym", "tests", "data")
        self.trace_set = glob.glob(f'{trace_dir}/**/*.json', recursive=True)
        # print("Trace set", self.trace_set)
        #Actions - actions can be from 0 to 1 (continuous actions)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float64)

        #state space - defined by 4 variables that go from 0 to 1 (continuous states)
        #States: receiving rate, average delay, loss ratio, latest prediction
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(4, 5), dtype=np.float64)

        self.state_dim = 4
        self.state_length = 5
        self.action_dim = 1

        self.list_of_packets = []

    def reset(self, training=True):
        self.gym_env = gym_file.Gym()

        if training:
            # self.current_trace = random.choice(self.trace_set)
            self.current_trace = "/home/dena/Documents/Gym_RTC/gym-example/traces/WIRED_900kbps.json"
        else:
            # self.current_trace = "/home/dena/Documents/Gym_RTC/gym-example/gym_folder/alphartc_gym/tests/data/WIRED_900kbs.json"
            self.current_trace = "/home/dena/Documents/Gym_RTC/gym-example/traces/WIRED_900kbps.json"
            # self.current_trace = "/home/dena/Documents/Gym_RTC/gym-example/traces/4G_500kbps.json"


        #Do the simulation with the current trace
        # logging.info("------------------------------------------")
        logging.info(f"{self.current_trace.split('/')[-1]}")
        self.gym_env.reset(trace_path=self.current_trace,
            report_interval_ms=self.step_time,
            duration_time_ms=0)

        #Initialize a new **empty** packet record
        self.packet_record = PacketRecord()

        self.state = torch.zeros(size=(self.state_dim, self.state_length))
        self.make_bandwidth_series()
        self.time_step_in_epoch = 0

        return self.state

    def make_bandwidth_series(self):
        #read bandwidth file and create series
        #index timestamps, values bandwidth
        with open(self.current_trace, "r") as f:
            d = json.load(f)
        df = pd.DataFrame(d["uplink"]["trace_pattern"])
        time = [0] + list(df["duration"].cumsum())
        capacities = [df["capacity"].iloc[0]] + list(df["capacity"])
        s = pd.Series(index=pd.to_datetime(time, unit="ms"), data=capacities)
        self.capacities = s.resample("200ms").ffill()
        # print(f"Num steps in one trace: {len(self.capacities)}")

    def get_bandwidth(self):
        #take the corresponding bandwidth for the current time step
        #code handled in step() for now
        pass


    def step(self, action):

        # action: log to linear
        bandwidth_prediction = log_to_linear(action)
        # bandwidth_prediction = 300000
        logging.info(f"Action scaled {action}")
        logging.info(f"Action original {int(bandwidth_prediction)}")
        self.bandwidth_prediction_class_var = int(bandwidth_prediction)
        # print("\n")
        # print("---------------RL AGENT STEP START --------------------")
        # print("Bandwidth prediction to input in next step: ", bandwidth_prediction)

        # run the action
        packet_list, done = self.gym_env.step(bandwidth_prediction)
        # logging.info(f"Packet list contains {packet_list}")
        # logging.info(f"Len packet list {len(packet_list)}")
        pkt_counter = 0
        for pkt in packet_list:
            packet_info = PacketInfo()
            packet_info.payload_type = pkt["payload_type"]
            packet_info.ssrc = pkt["ssrc"]
            packet_info.sequence_number = pkt["sequence_number"]
            packet_info.send_timestamp = pkt["send_time_ms"]
            packet_info.receive_timestamp = pkt["arrival_time_ms"]
            packet_info.padding_length = pkt["padding_length"]
            packet_info.header_length = pkt["header_length"]
            packet_info.payload_size = pkt["payload_size"]
            packet_info.bandwidth_prediction = bandwidth_prediction
            if pkt_counter == 0:
                packet_info.first_packet = True
            if pkt_counter == len(packet_list)-1:
                packet_info.last_packet = True
            pkt["first_packet"] = packet_info.first_packet
            pkt["last_packet"] = packet_info.last_packet
            self.list_of_packets.append(pkt)
            self.packet_record.on_receive(packet_info)
            pkt_counter += 1

        #calculate sending rate
        self.sending_rate = self.packet_record.calculate_sending_rate(interval=self.step_time)

        # calculate state
        states = []
        # print("Step time ", self.step_time)
        self.receiving_rate = self.packet_record.calculate_receiving_rate(interval=self.step_time)
        states.append(linear_to_log(self.receiving_rate))

        self.delay = self.packet_record.calculate_average_delay(interval=self.step_time)
        # print(f"Delay {self.delay} ms, appended in state vector: {min(self.delay/1000, 1)} s")
        states.append(min(self.delay/1000, 1))

        self.loss_ratio = self.packet_record.calculate_loss_ratio(interval=self.step_time)
        states.append(self.loss_ratio)

        latest_prediction = self.packet_record.calculate_latest_prediction()
        states.append(linear_to_log(latest_prediction))
        self.log_prediction = linear_to_log(latest_prediction)

        self.state = self.state.clone().detach()
        self.state = torch.roll(self.state, -1, dims=-1)

        #States normalized
        self.state[0, -1] = linear_to_log(self.receiving_rate)
        self.state[1, -1] = min(self.delay/1000, 1)
        self.state[2, -1] = self.loss_ratio
        self.state[3, -1] = linear_to_log(latest_prediction)

        #States not normalized
        # self.state[0, -1] = self.receiving_rate / 1000
        # self.state[1, -1] = self.delay
        # self.state[2, -1] = self.loss_ratio
        # self.state[3, -1] = latest_prediction

        # logging.info(
        #     f"Receiving rate state | Delay state | Loss ratio state | Latest prediction"
        #     f"      \n{states[0]},          {states[1]},            {states[2]},            {states[3]}  ")

        #Calculate bandwidth utilization
        self.current_time_step = self.time_step_in_epoch * self.step_time
        self.current_time_step = pd.to_datetime(self.current_time_step, unit = "ms")
        self.current_bandwidth = self.capacities.loc[self.current_time_step]

        # calculate reward
        #receiving rate - delay - loss_ratio
        reward = self.calculate_reward()
        # reward2 = states[0] - states[1] - states[2]

        self.time_step_in_epoch += 1

        # print("Sending rate: ", self.sending_rate, "bandwidth: ", self.current_bandwidth,
                  # "Receiving rate:", self.receiving_rate, "delay: ", self.delay)

        # logging.info(f"Reward: {reward}")
        # print("---------------RL AGENT STEP END --------------------")


        return self.state, reward, done, {}

    def calculate_reward(self):

        sending_rate = self.sending_rate / 1000
        bandwidth = self.current_bandwidth
        receiving_rate = self.receiving_rate / 1000
        delay = self.delay
        loss_ratio = self.loss_ratio
        bandwidth_util = receiving_rate / bandwidth

        # logging.info(f"Sending rate: {sending_rate}, Receiving rate: {receiving_rate}, "
        #              f"bandwidth: {bandwidth}, delay: {delay}, U: {round(receiving_rate / bandwidth, 2)}, "
        #              f"loss ratio: {loss_ratio}")

        #forbidden values - forse reward -1
        if (delay < 0) \
                or (loss_ratio > 1) \
                or (loss_ratio < 0) \
                or (bandwidth_util < 0) \
                or (bandwidth_util > 1):
            # logging.info(f"Unallowed values for delay, loss or bandwidth util - delay: {delay},"
            #              f"loss_ratio: {loss_ratio}, util: {bandwidth_util}")
            reward = -1
            return reward

        #force reward -1 if any of these is true
        elif (receiving_rate > bandwidth) or \
                (receiving_rate > sending_rate) or \
                (delay > 1000) or \
                (loss_ratio > 0.2):
            # logging.info("Conditioned reward -1")
            reward = -1
            return reward

        #force reward 1, if ALL of these are true
        elif (loss_ratio < 0.02) and \
                (delay < 30) and \
                (bandwidth_util > 0.9):
            # logging.info("Conditioned reward 1")
            reward = 1
            return reward

        # Covers cases for 0 <= utilization <= 1
        if (bandwidth_util >= 0) and (bandwidth_util <= 0.65):
            Ru = (1.538 * bandwidth_util) - 1
        elif (bandwidth_util > 0.65) and (bandwidth_util <= 1):
            Ru = -8.2 * ((bandwidth_util - 1) ** 2) + 1
        Ru = round(Ru, 4)

        # Covers cases for delay >= 0
        if (delay >= 0) and (delay <= 150):
            Rd = -0.00667 * delay + 1
        elif (delay > 150) and (delay <= 200):
            Rd = -0.02 * delay + 3
        elif (delay > 200):
            Rd = -1
        Rd = round(Rd, 4)

        # Covers cases for 0 <= l <= 1
        if (loss_ratio >= 0) and (loss_ratio <= 0.02):
            Rl = 1
        elif (loss_ratio > 0.02) and (loss_ratio <= 0.1):
            Rl = 156 * ((loss_ratio - 0.1) ** 2)
        elif (loss_ratio > 0.1) and (loss_ratio <= 0.2):
            Rl = 100 * ((loss_ratio - 0.2) ** 2) - 1
        elif (loss_ratio > 0.2) and (loss_ratio <= 1):
            Rl = -1
        Rl = round(Rl, 4)

        if loss_ratio > 0:
            reward = 0.333*Ru + 0.333*Rd + 0.333*Rl
            # print("Loss ratio larger than 0")
        else:
            reward = (2/5) * Ru + (2/5) * Rd + (1/5) * Rl

        # reward = linear_to_log(self.receiving_rate) - min(self.delay/1000, 1) - self.loss_ratio
        return reward


    def clear_list_of_packets(self):
        self.list_of_packets = []
