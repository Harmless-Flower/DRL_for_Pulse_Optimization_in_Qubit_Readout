from stable_baselines3.common.env_checker import check_env
from sim_numba import NumbaPulseEnv

env = NumbaPulseEnv()
check_env(env)