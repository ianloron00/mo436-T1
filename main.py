from dependencies import *
from environment import *
import gym

env = GameEnv(10, 5) 
# gym.make("env")

for _ in range(25000):
  obs = env.reset()
  done = False
  while (not done):
    action = env.action_space.sample()
    obs, reward, done, _ = env.step(action)
    obs = env.observation_space
    env.render()
env.close()
