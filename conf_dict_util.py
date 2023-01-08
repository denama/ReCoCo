

def conf_to_dict(conf_dict):
    
    d_splitted = conf_dict.split("_")
    alg = d_splitted[0]
    trace_name = d_splitted[1] + "_" + d_splitted[2]
    trace = "./traces/" + trace_name + ".json"
    step_time = int(d_splitted[3])
    
    if "reward_profile" in conf_dict:
        reward_profile = int(d_splitted[13])
    else:
        reward_profile = 0
        
    if "seed" in conf_dict:
        seed = int(d_splitted[15])
    else:
        seed = 0

    if d_splitted[5] == "True":
        delay_states = True
    else:
        delay_states = False
    if d_splitted[8] == "True":
        normalize_states = True
    else:
        normalize_states = False
    if d_splitted[10].split(".")[0] == "True":
        tuned = True
    else:
        tuned = False

    d_final = {
     'trace_path': trace,
     'trace_name': trace_name,
     'delay_states': delay_states,
     'normalize_states': normalize_states,
     'step_time': step_time,
     'alg': alg,
     'tuned': tuned,
     'reward_profile': reward_profile,
     'seed': seed,
    }
    
    return d_final


def dict_to_conf(d):
    conf_dict = "_".join([d["alg"], d["trace_name"], str(d["step_time"]), "delay", str(d["delay_states"]),
                 "norm_states", str(d["normalize_states"]), "tuned", str(d["tuned"]),
                ])
    
    if (not "reward_profile" in d.keys()) or (not "seed" in d.keys()):
        return conf_dict
    else:
        return "_".join([conf_dict, "reward_profile", str(d["reward_profile"]), "seed", str(d["seed"]),
                    ])