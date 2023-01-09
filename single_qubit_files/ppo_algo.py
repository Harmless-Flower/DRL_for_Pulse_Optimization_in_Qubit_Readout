from stable_baselines3 import PPO
import os
import time
from sim_numba import NumbaPulseEnv
import gym



models_dir = f"models/{int(time.time())}_PPO"
logdir = f"logs/{int(time.time())}"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(logdir):
    os.makedirs(logdir)

env = NumbaPulseEnv()
env.reset()

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logdir, device="cuda")
# Can also use CnnPolicy or MultiInputPolicy

TIMESTEPS = 1000
for i in range(1, 100):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"PPO")
    model.save(f"{models_dir}/{TIMESTEPS*i}")