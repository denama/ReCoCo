from trace_lists import low_bandwidth_traces, medium_bandwidth_traces, high_bandwidth_traces, train, test, train_v2, test_v2, train_v3, test_v3
from trace_lists import opennetlab_traces, norway_traces, ghent_traces, ny_traces
import pandas as pd

# Set the list of traces to train on: list_traces_curr
# If curriculum learning, set curr=True and set continue_training=True

# Running in Terminal 1
#-------------------------------------------------
# Training a curriculum
# Base model
conf_name_root = "random_all_data"
conf_name_curr_list = "random_all_data_curr2"
# Episode where to start the current curriculum
from_ep = 80
from_ep_for_curriculum_list = 80    # Will be different from from_ep only if running was stopped mid-20 episodes and has to be rerun
to_ep = 100
curr = True
first_iteration_curr = False
#--------------------------------------------------

# # # Running in Terminal 2
# #-------------------------------------------------
# # Training a curriculum
# # Base model
# conf_name_root = "random_train_80_20_v2"
# conf_name_curr_list = "random_train_80_20_v2_curr2"
# # Episode where to start the current curriculum
# from_ep = 80
# from_ep_for_curriculum_list = 80       # Will be different from from_ep only if running was stopped mid-20 episodes and has to be rerun
# to_ep = 100
# curr = True
# first_iteration_curr = False
# #--------------------------------------------------



# #Running in Terminal 1
# #-------------------------------------------------
# # Training a new model
# conf_name_root = "random_low_bandwidth"
# from_ep = 59
# to_ep = 75
# curr = False
# list_traces_curr = low_bandwidth_traces
# #--------------------------------------------------



num_timesteps = 80000

list_traces_curr_path = f'./curriculum_lists/{conf_name_curr_list}/ep{from_ep_for_curriculum_list}.pkl'
list_traces_curr = pd.read_pickle(list_traces_curr_path)
print(f"Taking list of traces from: {list_traces_curr_path}")


if curr == False:
    print("Curr is FALSE, you are training a new model")
    take_dir = f"{conf_name_root}"
    tensorboard_dir = f"./tensorboard_logs/{conf_name_root}/"
    save_subfolder = f"{conf_name_root}/"
else:
    if first_iteration_curr:
        take_dir = f"{conf_name_root}"
    else:            
        take_dir = f"{conf_name_root}/curr2/"
    tensorboard_dir = f"./tensorboard_logs/{conf_name_root}/curr2/"
    save_subfolder = f"{conf_name_root}/curr2/"


new_input_conf = dict(
tensorboard_dir = tensorboard_dir,
save_subfolder = save_subfolder,
suffix = conf_name_root,
num_timesteps = num_timesteps,

trace_set = list_traces_curr,
num_episodes = to_ep,
seed = 5,

continue_training = True,
model_num = from_ep * num_timesteps,
take_dir = take_dir
)



