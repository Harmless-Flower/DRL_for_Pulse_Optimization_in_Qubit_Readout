from sim import SingleQNKEnv

env = SingleQNKEnv()

episodes = 50

for episode in range(episodes):
    done = False
    obs = env.reset()
    while True:
        random_action = env.action_space.sample()
        print("action", random_action)
        obs, reward, done, info = env.step(random_action)
        print('reward: ', reward)