import torch


traces = ["./traces/WIRED_900kbps.json",
          "./traces/WIRED_35mbps.json",
          "./traces/WIRED_200kbps.json", 
          "./traces/4G_700kbps.json",
          "./traces/4G_3mbps.json",
          "./traces/4G_500kbps.json",
          "./big_trace/big_trace2.json",
           ]


input_conf = {
    "num_timesteps": 10000,
    "num_episodes": 15,
    "n_cores": 12,
    "save_dir": "./data_mp",
    "tensorboard_dir": f"./tensorboard_logs_mp",
    "rates_delay_loss_dir": f"./output_mp",
}


config_dict_grid = {
    "trace": traces,
    "delay_states": [True, False],
    "normalize_states": [True, False],
    "step_time": [200,400,600],
    "alg": ["SAC", "TD3", "PPO"],
    "tuned": [False, True], 
}


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