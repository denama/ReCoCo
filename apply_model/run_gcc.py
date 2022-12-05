import os
import pickle
from gym_folder.alphartc_gym import gym_file
from apply_model import BandwidthEstimator
from apply_model import BandwidthEstimator_hrcc
from apply_model import BandwidthEstimator_gcc

from collections import defaultdict

BWE_gcc = BandwidthEstimator_gcc.GCCEstimator()
bandwidth_prediction_gcc, _ = BWE_gcc.get_estimated_bandwidth()



gym_env = gym_file.Gym()

current_trace = "/home/dena/Documents/Gym_RTC/gym-example/traces/WIRED_900kbps.json"
# current_trace = "/home/dena/Documents/Gym_RTC/gym-example/traces/4G_500kbps.json"


def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]

list_of_traces = listdir_fullpath("../traces")

for current_trace in list_of_traces:
    print(current_trace)

    trace_name = current_trace.split("/")[-1].split(".")[0]
    print(trace_name)
    step_time = 200
    list_of_packets = []
    rates_delay_loss = defaultdict(list)
    rates_delay_loss["trace_name"] = trace_name

    #ON reset
    gym_env.reset(trace_path=current_trace,
                       report_interval_ms=step_time,
                       duration_time_ms=0)
    BWE_gcc.reset()

    # Initialize a new **empty** packet record
    # packet_record = PacketRecord()
    bandwidth_prediction_gcc, _ = BWE_gcc.get_estimated_bandwidth()

    #ON STEP
    for i in range(2000):
        packet_list, done = gym_env.step(bandwidth_prediction_gcc)
        for pkt in packet_list:
            BWE_gcc.report_states(pkt)

        bandwidth_prediction_gcc, _ = BWE_gcc.get_estimated_bandwidth()
        # print(bandwidth_prediction_gcc)
        # Calculate rate, delay, loss
        sending_rate = BWE_gcc.packet_record.calculate_sending_rate(interval=step_time)
        receiving_rate = BWE_gcc.packet_record.calculate_receiving_rate(interval=step_time)
        loss_ratio = BWE_gcc.packet_record.calculate_loss_ratio(interval=step_time)
        delay = BWE_gcc.packet_record.calculate_average_delay(interval=step_time)

        rates_delay_loss["bandwidth_prediction"].append(bandwidth_prediction_gcc)
        rates_delay_loss["sending_rate"].append(sending_rate)
        rates_delay_loss["receiving_rate"].append(receiving_rate)
        rates_delay_loss["delay"].append(delay)
        rates_delay_loss["loss_ratio"].append(loss_ratio)


        # print(f"Sending rate {sending_rate}, receiving rate {receiving_rate}, "
        #       f"prediction {bandwidth_prediction_gcc}, loss ratio {loss_ratio}")

        if done:
            print(f"DONE WITH THE TRACE. I reached i {i}")
            break

    with open(f"rates_delay_loss_gcc_{trace_name}.pickle", "wb") as f:
        pickle.dump(rates_delay_loss, f)

    print(rates_delay_loss)



# states = {
#     "send_time_ms": 100,
#     "arrival_time_ms": 400,
#     "payload_type": 125,
#     "sequence_number": 10,
#     "ssrc": 123,
#     "padding_length": 0,
#     "header_length": 120,
#     "payload_size": 1350
# }

# # Challenge example estimator - an RF model using a trained model "pretrained_model.pth"
# model_path_dena = "./model/ppo_2022_05_05_16_19_31_v2.pth"
# # model_path = "./model/pretrained_model.pth"
# model_path = "./model/ppo_2022_05_05_16_19_31_v2.pth"
# BWE = BandwidthEstimator.Estimator(model_path)
# bandwidth_prediction1 = int(BWE.get_estimated_bandwidth())
# print("Reality check step time: ", BWE.step_time)
# print("Prediction from challenge example estimator", bandwidth_prediction1)
# print("----------------------------------------------")
#
# # hrcc_model_path = './model/ppo_2021_07_25_04_57_11_with500trace.pth'
# hrcc_model_path = "./model/ppo_2022_05_05_16_19_31_v2.pth"
# BWE_hrcc = BandwidthEstimator_hrcc.Estimator(hrcc_model_path)
# bandwidth_prediction2 = BWE_hrcc.get_estimated_bandwidth()
# print("Reality check step time: ", BWE_hrcc.step_time)
# print("Prediction from HRCC estimator", bandwidth_prediction2)
# print("----------------------------------------------")
#
# BWE_gcc = BandwidthEstimator_gcc.GCCEstimator()
# bandwidth_prediction3, _ = BWE_gcc.get_estimated_bandwidth()
# print("Reality check state: ", BWE_gcc.state)
# print("Prediction fromGCC estimator", bandwidth_prediction3)

