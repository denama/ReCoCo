import os
import pickle
import sys
import time
import datetime


sys.path.append("/home/det_user/dmarkudova/RL_rtc/")
# sys.path.append("/home/dena/Documents/Gym_RTC/gym-example/")


from gym_folder.alphartc_gym import gym_file

from apply_model import BandwidthEstimator
from apply_model import BandwidthEstimator_hrcc
from apply_model import BandwidthEstimator_gcc

from collections import defaultdict
import multiprocessing


def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d) if ".json" in f]


def main_func(current_trace):

    BWE_gcc = BandwidthEstimator_gcc.GCCEstimator()
    bandwidth_prediction_gcc, _ = BWE_gcc.get_estimated_bandwidth()
    gym_env = gym_file.Gym()


    # print("Current trace:", current_trace)

    trace_name = current_trace.split("/")[-1].split(".")[0]
    # print("Trace name", trace_name)
    
    start = time.time()
    print(f"Trace {current_trace} starting at time: ", datetime.datetime.now())
    
    if os.path.exists(f"results_gcc/norway/rates_delay_loss_gcc_{trace_name}.pickle") or trace_name == "train_2011-02-14_1728CET":
        return
    
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
    for i in range(80000):
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


        if done:
            end = time.time()
            print(f"DONE WITH THE TRACE. I reached i {i} in {end-start} seconds")
            break

    with open(f"results_gcc/norway/rates_delay_loss_gcc_{trace_name}.pickle", "wb") as f:
        pickle.dump(rates_delay_loss, f)



if __name__ == "__main__":
    
    list_of_traces = listdir_fullpath("../new_data/Norway_3G_data_json/")
    
    print(f"Doing {len(list_of_traces)} traces .... ")

    n_cores = 50
    pool = multiprocessing.Pool(processes=n_cores)
    pool.map(main_func, list_of_traces)