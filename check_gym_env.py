
from stable_baselines3.common.env_checker import check_env

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from rtc_env_sb import GymEnv
from stable_baselines3 import PPO, A2C
import logging
from collections import defaultdict
import pickle

# logging.basicConfig(filename='logs/check_gym_env.log', level=logging.INFO, filemode='w')
# logging.info('Started apply model')


# env = GymEnv()
# check_env(env, warn=True)


#1 Train an agent

# env = GymEnv()
# env = make_vec_env(lambda: env, n_envs=1)
# model = A2C('MlpPolicy', env, verbose=1).learn(5000)

# print("-----------------------")
# print("DONE WITH TRAINING")
# print("-----------------------")

#Load a saved agent
save_dir = "./data"
model = PPO.load(save_dir + "/PPO/400000")
env = GymEnv()
# env = make_vec_env(lambda: env, n_envs=1)

# Test the trained agent
rates_delay_loss = defaultdict(list)


obs = env.reset()

rates_delay_loss = {"trace": env.current_trace,
                                   "bandwidth_prediction": [],
                                   "sending_rate": [],
                                   "receiving_rate": [],
                                   "delay": [],
                                   "loss_ratio": [],
                                   "log_prediction": [],
                                   "reward": []
                                   }

n_steps = 1000
for step in range(n_steps):
  action, _ = model.predict(obs, deterministic=True)

  # print("Step {}".format(step + 1))
  # print("Action: ", action)
  obs, reward, done, info = env.step(action)
  # print('obs=', obs, 'reward=', reward, 'done=', done)
  # print("action", action)

  rates_delay_loss["bandwidth_prediction"].append(env.bandwidth_prediction_class_var)
  rates_delay_loss["sending_rate"].append(env.sending_rate)
  rates_delay_loss["receiving_rate"].append(env.receiving_rate)
  rates_delay_loss["delay"].append(env.delay)
  rates_delay_loss["loss_ratio"].append(env.loss_ratio)
  rates_delay_loss["log_prediction"].append(float(env.log_prediction))
  rates_delay_loss["reward"].append(reward)

  if done:
    # Note that the VecEnv resets automatically
    # when a done signal is encountered
    print("Goal reached!", "reward=", reward)

    with open("rates_delay_loss_sb.pickle", "wb") as f:
        pickle.dump(rates_delay_loss, f)

    break


