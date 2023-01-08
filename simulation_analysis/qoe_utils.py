import json
import pandas as pd
import numpy as np

def make_bandwidth_series(trace_path, step_time):
    
    #read bandwidth file and create series
    #index timestamps, values bandwidth
    with open(trace_path, "r") as f:
        d = json.load(f)
    df = pd.DataFrame(d["uplink"]["trace_pattern"])
    time = [0] + list(df["duration"].cumsum())
    capacities = [df["capacity"].iloc[0]] + list(df["capacity"])
    s = pd.Series(index=pd.to_datetime(time, unit="ms"), data=capacities)
    capacities = s.resample(f"{step_time}ms").bfill()
    return capacities


def get_QoE_rr(l_rate, l_rate_gcc, REAL_NAME, time_step, test=False):

    metric = "receiving_rate"
    m=9
    key = "./traces/" + REAL_NAME + ".json"
    trace_path = "../traces/" + REAL_NAME + ".json"
    capacities = make_bandwidth_series(trace_path, time_step)
    
    if test:
        df_rate = pd.DataFrame(l_rate[key][metric], columns=[metric])
    else:
        df_rate = pd.DataFrame(l_rate[key][m][metric], columns=[metric])
    
    df_rate_gcc = pd.DataFrame(l_rate_gcc[metric], columns=[metric])
    df_rate = df_rate.join(df_rate_gcc, rsuffix="_gcc")

    t = np.arange(len(df_rate))*time_step
    df_rate["time"] = t
    df_rate["time"] = pd.to_datetime(df_rate["time"], unit="ms")
    df_rate.set_index("time", inplace=True)

    receiving_rate_kbps = df_rate["receiving_rate"]/1000
    # capacities_kbps = capacities.replace(0, 100000000)
    capacities_kbps = capacities.replace(0, 0.001)

    rr_kbps_gcc = df_rate["receiving_rate_gcc"]/1000

    U = receiving_rate_kbps / capacities_kbps
    U_gcc = rr_kbps_gcc / capacities_kbps

    U2 = U.clip(0,1)
    U_gcc2 = U_gcc.clip(0,1)
    
    return U2, U_gcc2


def get_QoE_delay(l_rate, l_rate_gcc, REAL_NAME, time_step, test=False):

    metric = "delay"
    m=9
    key = "./traces/" + REAL_NAME + ".json"
    
    if test:
        df_rate = pd.DataFrame(l_rate[key][metric], columns=[metric])
    else:
        df_rate = pd.DataFrame(l_rate[key][m][metric], columns=[metric])

    df_rate_gcc = pd.DataFrame(l_rate_gcc[metric], columns=[metric])
    df_rate = df_rate.join(df_rate_gcc, rsuffix="_gcc")

    t = np.arange(len(df_rate))*time_step
    df_rate["time"] = t
    df_rate["time"] = pd.to_datetime(df_rate["time"], unit="ms")
    df_rate.set_index("time", inplace=True)

    delay = df_rate["delay"]
    delay_gcc = df_rate["delay_gcc"]
    
    d_max = delay.max()
    d_min = delay.min()
    d_95 = delay.quantile(0.95)

    d_max_gcc = delay_gcc.max()
    d_min_gcc = delay_gcc.min()
    d_95_gcc = delay_gcc.quantile(0.95)
    
    qoe_delay = 100*(d_max-d_95)/(d_max - d_min)
    qoe_delay_gcc = 100*(d_max_gcc - d_95_gcc) / (d_max_gcc - d_min_gcc)
    
    return delay, delay_gcc, qoe_delay, qoe_delay_gcc


def get_QoE_losses(l_rate, l_rate_gcc, REAL_NAME, time_step, test=False):

    metric = "loss_ratio"
    m=9
    key = "./traces/" + REAL_NAME + ".json"

    if test:
        df_rate = pd.DataFrame(l_rate[key][metric], columns=[metric])
    else:
        df_rate = pd.DataFrame(l_rate[key][m][metric], columns=[metric])
    
    df_rate_gcc = pd.DataFrame(l_rate_gcc[metric], columns=[metric])
    df_rate = df_rate.join(df_rate_gcc, rsuffix="_gcc")

    t = np.arange(len(df_rate))*time_step
    df_rate["time"] = t
    df_rate["time"] = pd.to_datetime(df_rate["time"], unit="ms")
    df_rate.set_index("time", inplace=True)

    loss_ratio = df_rate["loss_ratio"]
    loss_ratio_gcc = df_rate["loss_ratio_gcc"]
    
    L = loss_ratio.mean()
    L_gcc = loss_ratio_gcc.mean()
    
    qoe_losses = 100*(1-L)
    qoe_losses_gcc = 100*(1-L_gcc)
    
    return loss_ratio, loss_ratio_gcc, qoe_losses, qoe_losses_gcc


def get_reward(l_rate, REAL_NAME, test=False):

    metric = "reward"
    m=9
    key = "./traces/" + REAL_NAME + ".json"

    if test:
        df_rate = pd.DataFrame(l_rate[key][metric], columns=[metric])
    else:
        df_rate = pd.DataFrame(l_rate[key][m][metric], columns=[metric])
    

    cum_reward = df_rate["reward"].sum()
    avg_reward = df_rate["reward"].mean()

    
    return cum_reward, avg_reward