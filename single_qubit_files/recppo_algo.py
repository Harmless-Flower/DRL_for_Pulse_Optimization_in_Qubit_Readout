import numpy as np
import os
import time

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.evaluation import evaluate_policy
from sim_numba import NumbaPulseEnv

models_dir = f"models/{int(time.time())}_RecPPO"
logdir = f"logs/{int(time.time())}"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(logdir):
    os.makedirs(logdir)

env = NumbaPulseEnv()

model = RecurrentPPO("MlpLstmPolicy", env, verbose=1, tensorboard_log=logdir)

TIMESTEPS = 10000
for i in range(1, 1000):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"RecPPO")
    model.save(f"{models_dir}/{TIMESTEPS*i}")