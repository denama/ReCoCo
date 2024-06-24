#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging

import numpy as np
from collections import defaultdict


class PacketRecord:
    # feature_interval can be modified
    def __init__(self, base_delay_ms=0):
        self.base_delay_ms = base_delay_ms
        self.reset()

    def reset(self):
        self.packet_num = 0
        self.packet_list = []
        self.last_seqNo = {}
        self.timer_delta = None  # ms
        self.min_seen_delay = self.base_delay_ms  # ms
        # ms, record the rtime of the last packet in last interval,
        self.last_interval_rtime = None
        self.interpolated_packet_sizes = defaultdict(list)

    def clear(self):
        self.packet_num = 0
        if self.packet_list:
            self.last_interval_rtime = self.packet_list[-1]['timestamp']
        self.packet_list = []

    def on_receive(self, packet_info):
        assert (len(self.packet_list) == 0
                or packet_info.receive_timestamp
                >= self.packet_list[-1]['timestamp']), \
            "The incoming packets receive_timestamp disordered"

        # Calculate the loss count
        loss_count = 0

        if packet_info.ssrc in self.last_seqNo:
            loss_count = max(0,
                packet_info.sequence_number - self.last_seqNo[packet_info.ssrc] - 1)
            self.interpolated_packet_sizes[packet_info.ssrc].append(packet_info.payload_size)
            # print(self.interpolated_packet_sizes)
        self.last_seqNo[packet_info.ssrc] = packet_info.sequence_number

        # Calculate packet delay
        if self.timer_delta is None:
            # shift delay of the first packet to base delay
            self.timer_delta = self.base_delay_ms - (packet_info.receive_timestamp - packet_info.send_timestamp)
        self.timer_delta = 0
        delay = self.timer_delta + packet_info.receive_timestamp - packet_info.send_timestamp
        # logging.info(f"Timer delta {self.timer_delta} Receive timestamp {packet_info.receive_timestamp} Send timestamp {packet_info.send_timestamp}")
        self.min_seen_delay = min(delay, self.min_seen_delay)
        # logging.info(f"Recevied - sent {packet_info.receive_timestamp - packet_info.send_timestamp}")
        # logging.info(f"Delay {delay}")

        # Check the last interval rtime
        if self.last_interval_rtime is None:
            self.last_interval_rtime = packet_info.receive_timestamp

        # Record result in current packet
        packet_result = {
            'timestamp': packet_info.receive_timestamp,  # ms
            'delay': delay,  # ms
            'payload_byte': packet_info.payload_size,  # B
            'loss_count': loss_count,  # p
            'bandwidth_prediction': packet_info.bandwidth_prediction  # bps
        }
        self.packet_list.append(packet_result)
        self.packet_num += 1

    def _get_result_list(self, interval, key):
        if self.packet_num == 0:
            return []

        result_list = []
        if interval == 0:
            interval = self.packet_list[-1]['timestamp'] - self.last_interval_rtime
        start_time = self.packet_list[-1]['timestamp'] - interval
        index = self.packet_num - 1
        while index >= 0 and self.packet_list[index]['timestamp'] > start_time:
            result_list.append(self.packet_list[index][key])
            index -= 1

        return result_list

    def calculate_average_delay(self, interval=0):
        '''
        Calulate the average delay in the last interval time,
        interval=0 means based on the whole packets
        The unit of return value: ms
        '''
        delay_list = self._get_result_list(interval=interval, key='delay')
        # print(delay_list)
        if delay_list:
            # print("Delay list ", delay_list, "base delay ", self.base_delay_ms)
            return np.mean(delay_list) - self.base_delay_ms
        else:
            return 0

    def calculate_loss_ratio(self, interval=0):
        '''
        Calulate the loss ratio in the last interval time,
        interval=0 means based on the whole packets
        The unit of return value: packet/packet
        '''
        loss_list = self._get_result_list(interval=interval, key='loss_count')
        if loss_list:
            loss_num = np.sum(loss_list)
            received_num = len(loss_list)
            # print(f"Num lost {loss_num}, num received {received_num}, loss rate {loss_num / (loss_num + received_num)}")
            return loss_num / (loss_num + received_num)
        else:
            return 0

    def calculate_receiving_rate(self, interval=0):
        '''
        Calulate the receiving rate in the last interval time,
        interval=0 means based on the whole packets
        interval is in ms
        The unit of return value: bps
        '''
        received_size_list = self._get_result_list(interval=interval, key='payload_byte')
        if received_size_list:
            received_nbytes = np.sum(received_size_list)
            if interval == 0:
                print("Interval is 0")
                interval = self.packet_list[-1]['timestamp'] - self.last_interval_rtime
            rate_output_bps = 1000 * received_nbytes * 8 / interval
            # logging.info(f"Packet sizes list {received_size_list}")
            # logging.info(f"Received {received_nbytes} Bytes in {interval} ms, so receiving rate is {rate_output_bps:.2f} bps or {(rate_output_bps / 1000):.2f} kbps")
            return rate_output_bps
        else:
            return 0

    def calculate_sending_rate(self, interval=0):

        loss_list = self._get_result_list(interval=interval, key='loss_count')
        bytes_list = self._get_result_list(interval=interval, key='payload_byte')
        if bytes_list:
            loss_count = np.sum(loss_list)
            average_bytes = round(np.mean(bytes_list))
            nbytes = np.sum(bytes_list)
            missing_nbytes = loss_count * average_bytes
            total_nbytes = nbytes + missing_nbytes

            # if loss_count > 0:
            #     logging.info(f"Received: {nbytes}, missing {missing_nbytes}, total {total_nbytes}")

            # TODO interpolate better using ssrc
            # sum of bytes list is sometimes smaller than nbytes_sent - because packet_list is ahead in time

            if interval == 0:
                print("Interval for sending rate is 0: ", interval)
                return None
            rate = (total_nbytes * 8 * 1000) / interval
            return rate
        else:
            return 0


    def calculate_latest_prediction(self):
        '''
        The unit of return value: bps
        '''
        if self.packet_num > 0:
            return self.packet_list[-1]['bandwidth_prediction']
        else:
            return 0