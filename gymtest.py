import gym
import nonstationary_pendulum

gym.envs.register(
    id='NonstationaryPendulum-v0',
    entry_point = 'nonstationary_pendulum:NonstationaryPendulumEnv',
    kwargs = {'drift_speed':0.001, 'drift_type':'oscillating', 'schedule':'unsupervised'}
)
env = gym.make('NonstationaryPendulum-v0')
for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()
