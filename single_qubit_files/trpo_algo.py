import numpy as np
from sb3_contrib import TRPO
import time
import os
from sim_numba import NumbaPulseEnv

env = NumbaPulseEnv()

models_dir = f"models/{int(time.time())}_TRPO"
logdir = f"logs/{int(time.time())}"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(logdir):
    os.makedirs(logdir)

model = TRPO("MlpPolicy", env, verbose=1, tensorboard_log=logdir)
TIMESTEPS = 10000
for i in range(1, 1000):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"TRPO")
    model.save(f"{models_dir}/{TIMESTEPS*i}")