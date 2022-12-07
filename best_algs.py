#Best models per tracefile, by visual inspection of reward (plot_avg_reward.ipynb)

from collections import defaultdict
best_models_dict = defaultdict(dict)

best_models_dict["WIRED_900kbps"][200] = "SAC_WIRED_900kbps_200_delay_False_norm_states_True_tuned_True"
best_models_dict["WIRED_900kbps"][400] = "TD3_WIRED_900kbps_400_delay_False_norm_states_True_tuned_False"
best_models_dict["WIRED_900kbps"][600] = "TD3_WIRED_900kbps_600_delay_False_norm_states_True_tuned_False"

best_models_dict["WIRED_200kbps"][200] = "TD3_WIRED_200kbps_200_delay_False_norm_states_True_tuned_False"
best_models_dict["WIRED_200kbps"][400] = "TD3_WIRED_200kbps_400_delay_False_norm_states_True_tuned_True"
best_models_dict["WIRED_200kbps"][600] = "TD3_WIRED_200kbps_600_delay_False_norm_states_True_tuned_False"

#Check these out on tensorboard
best_models_dict["WIRED_35mbps"][200] = "TD3_WIRED_35mbps_200_delay_False_norm_states_True_tuned_True"
best_models_dict["WIRED_35mbps"][400] = "TD3_WIRED_35mbps_400_delay_True_norm_states_True_tuned_False"
best_models_dict["WIRED_35mbps"][600] = "TD3_WIRED_35mbps_600_delay_True_norm_states_True_tuned_False"

best_models_dict["4G_700kbps"][200] = "TD3_4G_700kbps_200_delay_True_norm_states_True_tuned_False"
best_models_dict["4G_700kbps"][400] = "TD3_4G_700kbps_400_delay_True_norm_states_True_tuned_True"
best_models_dict["4G_700kbps"][600] = "TD3_4G_700kbps_600_delay_True_norm_states_True_tuned_False"

best_models_dict["4G_3mbps"][200] = "TD3_4G_3mbps_200_delay_False_norm_states_True_tuned_True"
best_models_dict["4G_3mbps"][400] = "TD3_4G_3mbps_400_delay_True_norm_states_True_tuned_False"
best_models_dict["4G_3mbps"][600] = "SAC_4G_3mbps_600_delay_True_norm_states_True_tuned_False"

best_models_dict["4G_500kbps"][200] = "TD3_4G_500kbps_200_delay_False_norm_states_True_tuned_True"
best_models_dict["4G_500kbps"][400] = "TD3_4G_500kbps_400_delay_True_norm_states_True_tuned_False"
best_models_dict["4G_500kbps"][600] = "SAC_4G_500kbps_600_delay_True_norm_states_True_tuned_False"
