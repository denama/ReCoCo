import os
import pickle
from gym_folder.alphartc_gym import gym_file
from apply_model import BandwidthEstimator
from apply_model import BandwidthEstimator_hrcc

from collections import defaultdict


hrcc_model_path = "./model/ppo_2021_07_25_04_57_11_with500trace.pth"


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
    # BWE_hrcc.reset()
    BWE_hrcc = BandwidthEstimator_hrcc.Estimator(hrcc_model_path)

    # Initialize a new **empty** packet record
    # packet_record = PacketRecord()
    bandwidth_prediction_hrcc = BWE_hrcc.get_estimated_bandwidth()
    # print(bandwidth_prediction_hrcc)


    #ON STEP
    for i in range(2000):
        packet_list, done = gym_env.step(bandwidth_prediction_hrcc)
        for pkt in packet_list:
            BWE_hrcc.report_states(pkt)

        bandwidth_prediction_hrcc = BWE_hrcc.get_estimated_bandwidth()
        print("Final banwidth prediction: ", bandwidth_prediction_hrcc)
        # Calculate rate, delay, loss
        sending_rate = BWE_hrcc.packet_record.calculate_sending_rate(interval=step_time)
        receiving_rate = BWE_hrcc.receiving_rate
        loss_ratio1 = BWE_hrcc.packet_record.calculate_loss_ratio(interval=step_time)
        loss_ratio = BWE_hrcc.loss_ratio
        delay = BWE_hrcc.delay

        if loss_ratio1 != loss_ratio:
            print("Problem with loss ratio")
            exit()

        rates_delay_loss["bandwidth_prediction"].append(bandwidth_prediction_hrcc)
        rates_delay_loss["sending_rate"].append(sending_rate)
        rates_delay_loss["receiving_rate"].append(receiving_rate)
        rates_delay_loss["delay"].append(delay)
        rates_delay_loss["loss_ratio"].append(loss_ratio)


        # print(f"Sending rate {sending_rate}, receiving rate {receiving_rate}, "
        #       f"prediction {bandwidth_prediction_hrcc}, loss ratio {loss_ratio}")

        if done:
            print(f"DONE WITH THE TRACE. I reached i {i}")
            break

    with open(f"results_hrcc/rates_delay_loss_hrcc_{trace_name}_bla.pickle", "wb") as f:
        pickle.dump(rates_delay_loss, f)



