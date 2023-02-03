## About
**ReCoCo** is a Reinforcement-learning based Congestion Control algorithm for Real-time applications (video conferencing, cloud gaming).

## System overview
![Schema of the system](scheme/scheme_rl_system.drawio3.drawio.png)

## Scripts

- train_test_one_trace: train on one trace for 10000 samples, 10 times (fake episodes). Every episode save model, test and record reward (rates_delay_loss[trace][m]). You specify the trace and the algorithm and the number of environments for vectorization (note: TD3 does not support vectorization)
- mp+train_test_one_trace: try different parameters and run with multiprocessing
- train_test_one_by_one: train on one trace for X timesteps, Y times (fake episodes). Then save model and continue training with another tracefile

- train_vec_env: just training
- test_env: just testing - load any model you want
- main_apply_sb: to test policy with stable_baselines evaluator function

- check_gym_env_original: legacy, keep for debugging stuff
- tensorboard_trial: legacy, keep for debugging stuff

- rtc_env_sb: environment exactly following Gym guidelines, to use with stable_baselines


- plotting_sb: to explore plotting, finish

- rtc_env: legacy, works with old main code (without stable baselines)
- main: legacy, use PPO implementation in deep_rl/ and rtc_env
- main_apply_model: legacy, test model from PPO implementation in deep_rl/ and rtc_env

