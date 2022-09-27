#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import random
import numpy as np
import glob
import logging

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


def liner_to_log(value):
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


class GymEnv:
    def __init__(self, step_time=100):
        self.gym_env = None     
        self.step_time = step_time
        #TODO train with whole trace set

        trace_dir = os.path.join(os.path.dirname(__file__), "traces")
        # trace_dir = os.path.join(os.path.dirname(__file__), "gym_folder", "alphartc_gym", "tests", "data")
        self.trace_set = glob.glob(f'{trace_dir}/**/*.json', recursive=True)
        # print("Trace set", self.trace_set)
        #Actions - actions can be from 0 to 1 (continuous actions)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float64)
        #state space - defined by 4 variables that go from 0 to 1 (continuous states)
        #States: receiving rate, average delay, loss ratio, latest prediction
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0, 1.0]),
            dtype=np.float64)

        self.list_of_packets = []

    def reset(self, training=True):
        self.gym_env = gym_file.Gym()

        if training:
            self.current_trace = random.choice(self.trace_set)
        else:
            self.current_trace = "/home/dena/Documents/Gym_RTC/gym-example/gym_folder/alphartc_gym/tests/data/WIRED_900kbs.json"
            # self.current_trace = "/home/dena/Documents/Gym_RTC/gym-example/traces/4G_700kbps.json"


        #Do the simulation with the current trace
        # logging.info("------------------------------------------")
        logging.info(f"{self.current_trace.split('/')[-1]}")
        self.gym_env.reset(trace_path=self.current_trace,
            report_interval_ms=self.step_time,
            duration_time_ms=0)

        #Initialize a new **empty** packet record
        self.packet_record = PacketRecord()
        # self.packet_record.reset()

        return [0.0, 0.0, 0.0, 0.0]


    def step(self, action):
        # action: log to linear
        bandwidth_prediction = log_to_linear(action)
        # bandwidth_prediction = 300000
        # logging.info(f"Action in linear {int(bandwidth_prediction)}")
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
        self.sending_rate = self.packet_record.calculate_sending_rate(packet_list, interval=self.step_time)

        # calculate state
        states = []
        # print("Step time ", self.step_time)
        self.receiving_rate = self.packet_record.calculate_receiving_rate(interval=self.step_time)

        # print("Linear to log for receiving rate: ")
        if packet_list:
            self.time = packet_list[-1]["arrival_time_ms"]
        else:
            self.time = np.nan
        # print("Arrival time of last packet: ", self.time)
        states.append(liner_to_log(self.receiving_rate))
        self.delay = self.packet_record.calculate_average_delay(interval=self.step_time)
        # print(f"Delay {self.delay} ms, appended in state vector: {min(self.delay/1000, 1)} s")
        states.append(min(self.delay/1000, 1))
        self.loss_ratio = self.packet_record.calculate_loss_ratio(interval=self.step_time)
        states.append(self.loss_ratio)
        latest_prediction = self.packet_record.calculate_latest_prediction()
        # print("Latest prediction: ", latest_prediction)
        # print("Linear to log for rate prediction: ")
        states.append(liner_to_log(latest_prediction))
        self.log_prediction = liner_to_log(latest_prediction)

        # logging.info(
        #     f"Receiving rate state | Delay state | Loss ratio state | Latest prediction"
        #     f"      \n{states[0]},          {states[1]},            {states[2]},            {states[3]}  ")

        # calculate reward
        #receiving rate - delay - loss_ratio
        reward = states[0] - states[1] - states[2]
        logging.info(f"Reward: {reward}")
        # print("---------------RL AGENT STEP END --------------------")


        return states, reward, done, {}


    def clear_list_of_packets(self):
        self.list_of_packets = []
