import torch
import itertools
import pandas as pd

traces = []

input_conf = {
    "num_timesteps": 10000,
    "num_episodes": 10,
    "n_cores": 12,
    "save_dir": "./data_mp",
    "tensorboard_dir": f"./tensorboard_logs_mp",
    "rates_delay_loss_dir": f"./output_mp",
    # "seed": 22,
}


config_dict_grid = {
    "trace_path": traces,
    "delay_states": [False, True],
    "normalize_states": [True],
    "step_time": [200],
    "alg": ["SAC", "TD3"],
    "tuned": [False, True],
    "reward_profile": [0,1,2,3,4],
}

#If you want to do all combinations from config_dict_grid
# keys, values = zip(*config_dict_grid.items())
# permutation_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]

#Your own permutation dict
pickle_perm_dicts = "./diff_random_seeds.pkl"
permutation_dicts = pd.read_pickle(pickle_perm_dicts)


#TD3
hyperparams_TD3 = dict(
    policy = "MlpPolicy",
    gamma = 0.98,
    learning_starts = 10000,
)

#SAC
hyperparams_SAC =  dict(
    policy="MlpPolicy",
    learning_rate=3e-4,
    buffer_size = 50000,
    learning_starts = 0,
    batch_size = 512,
    tau = 0.01,
    gamma = 0.9999,
    train_freq=32,
    gradient_steps=32,
    ent_coef=0.1,
    use_sde = True,
    policy_kwargs=dict(
        log_std_init=-3.67,
        net_arch=[64, 64]
    )
)

#PPO
hyperparams_PPO =  dict(
    policy="MlpPolicy",
    learning_rate=7.77e-05,
    n_steps=8,
    batch_size=8,
    n_epochs=2,
    gamma=0.9999,
    gae_lambda=0.9,
    clip_range=0.1,
    ent_coef=0.00429,
    vf_coef=0.19,
    max_grad_norm=5,
    use_sde=True,
    policy_kwargs=dict(
            log_std_init=-3.29,
            ortho_init=False,
            activation_fn=torch.nn.ReLU,
            )
)