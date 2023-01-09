from sim_numba import NumbaPulseEnv
import numpy as np
from sb3_contrib import RecurrentPPO

env = NumbaPulseEnv()
model = RecurrentPPO.load("models/")

obs = env.reset()
lstm_states = None
num_envs = 1

#Episode start signals are used to reset the lstm states
episode_starts = np.ones((num_envs,), dtype=bool)
action, lstm_states = model.predict(obs, state=lstm_states, episode_starts=episode_starts, deterministic=True)
env.grapher(action)