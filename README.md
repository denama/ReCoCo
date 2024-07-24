## About
**ReCoCo** is a Reinforcement-learning based Congestion Control algorithm for Real-time applications (video conferencing, cloud gaming).

## System overview
The system at runtime:

![The system at runtime](scheme/congestion_control.png)

The system at training time:

![Schema of the system](scheme/scheme_rl_system.drawio3.drawio.png)

## Scripts
Here is an outline of the main scripts used to train and test the various models:

- *test_env_github*: employs testing - takes an input argument - model name from folder */final_models*

- *conf_dict_new*: configure all parameters to start training (e.g. number of timesteps, algorithm, saving paths and other parameters)
- *trace_lists*: specifies different groupings of traces in lists for easier use in the other scripts (e.g. norway_traces, ghent_traces, low_bandwidth_traces)

- *train_new_data*: reads the configuration from *conf_dict_new* and employs training
- *test_env_non_testable_traces*: test models without running GCC in parallel, for the traces that do not support GCC
- *rtc_env*: RL environment (exactly following Gym guidelines)

## Quick guide
First, create a virtual environment in Python and install the packages using the *requirements.txt* file.
Then you can test the various models from the folder *final_models* by running *test_env_github.py* with the name of the model as an argument. This tests the model on *all* available traces in the *new_data* folder. To specify a specific list of traces to test on, modify the *traces* parameter in *test_env_github.py*. You can use predefined trace lists from *trace_lists.py*.

The output is a pickle file that is saved in *output_github*, which contains a dictionary with data such as bandwidth prediction, sending rate, receiving rate, delay, loss ratio and reward, for each time step in the simulation.


## Paper
If you use this code, please cite the following paper:

@inproceedings{markudova2023recoco,  
  title={ReCoCo: Reinforcement learning-based Congestion control for Real-time applications},  
  author={Markudova, Dena and Meo, Michela},  
  booktitle={2023 IEEE 24th International Conference on High Performance Switching and Routing (HPSR)},  
  pages={68--74},  
  year={2023},  
  organization={IEEE}  
}

