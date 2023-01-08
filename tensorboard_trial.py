import argparse
import numpy as np

from stable_baselines3 import A2C, PPO, SAC, TD3
from stable_baselines3.common.noise import NormalActionNoise
from legacy.rtc_env_sb import GymEnv

parser = argparse.ArgumentParser(description="Choose RL parameters")
parser.add_argument("-alg", "--algorithm", help="Choose algorithm", choices=['A2C', 'PPO', 'SAC', 'TD3'], default=None, required=True)

args = parser.parse_args()
print("Args: ", args)

alg = args.algorithm
print("Alg type, should be string: ", type(alg))
# exit(0)

# alg = "SAC"
#"A2C", "PPO", "SAC", "TD3" 

model_save_dir = f"./data/{alg}"
tensorboard_dir = "./tensorboard_logs/"

env = GymEnv()
env.reset()

if alg == "A2C":
    model = A2C('MlpPolicy', env, verbose=1, tensorboard_log=tensorboard_dir)
elif alg == "PPO":
    learning_rate = 0.001
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=learning_rate, tensorboard_log=tensorboard_dir)
elif alg == "SAC":
    learning_rate = 0.001
    model = SAC("MlpPolicy", env, verbose=1, learning_rate=learning_rate, tensorboard_log=tensorboard_dir)
elif alg == "TD3":
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    model = TD3("MlpPolicy", env, action_noise=action_noise, verbose=1, tensorboard_log=tensorboard_dir)
    

TIMESTEPS = 10000
for i in range(15):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=alg)
    model.save(f"{model_save_dir}/{TIMESTEPS*i}")