from stable_baselines3.common.env_checker import check_env
from sim import SingleQNKEnv

env = SingleQNKEnv()
check_env(env)