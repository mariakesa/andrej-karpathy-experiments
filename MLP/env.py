import gymnasium as gym
#env = gym.make("LunarLander-v2", render_mode="human")
env=gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True,render_mode="human")
env=observation, info = env.reset()
from gymnasium.spaces import Discrete
action_space = Discrete(4)

for _ in range(1000):
    #print(env)
    #action_space=range(0,4)
    action = action_space.sample()  # agent policy that uses the observation and info
    print(action)
    print(env)
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()