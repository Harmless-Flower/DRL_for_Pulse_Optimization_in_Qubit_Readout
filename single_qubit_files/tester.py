from sim_numba import NumbaPulseEnv
from stable_baselines3 import PPO, A2C
from sb3_contrib import TRPO, RecurrentPPO
import numpy as np

env = NumbaPulseEnv()

models_dir = "models"
model_path = f"{models_dir}/1673195597_TRPO/700000.zip"
model = TRPO.load(model_path, env=env)
obs = env.reset()
action, _ = model.predict(obs)

env.grapher(action)






T_LENGTH = 100

action = np.ones(2*(T_LENGTH + 1) + 1, dtype=np.float32)
action[int(T_LENGTH/2): T_LENGTH + 1] = -1
action[T_LENGTH + 1 + int(T_LENGTH/2):-1] = -1