from stable_baselines3 import PPO
from sim import SingleQNKEnv

env = SingleQNKEnv()
env.reset()

models_dir = "models/PPO"

model_path = f"{models_dir}/XXX.zip"

model = PPO.load(model_path, env=env)

episodes = 5

for ep in range(episodes):
    obs = env.reset()
    done = False
    while not done:
        env.render()
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)