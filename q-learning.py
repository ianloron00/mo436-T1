from os import stat
import gym
import numpy as np
import matplotlib.pyplot as plt
import random
from display import plot_rewards, plot_victories
from environment import GameEnv 
import pickle
from dependencies import *

env = GameEnv(10, 5) 

ALPHA = 0.1
GAMMA = 0.95
EPISODES = 25_000
EPSILON = 0.99
# very small value
# eps_decay_value = EPSILON/(EPISODES//2) 
EPS_DECAY = 0.9998

N_MAX_STEPS = 300
SHOW_EVERY = 3000

NAME_TABLE = 'q_learning_table'

start_q_table =  None # or name. e.g., 'q_learning_table.pickle'

# q_table shape: [((2*height, 2*width), (2*height, 2*width))][4]
def initialize_q_table(env):
  q_table = {}
  h = env.height    
  w = env.width
  for x1 in range(-w+1, w):
    for y1 in range(-h+1, h):
      for x2 in range(-w+1, w):
        for y2 in range(-h+1, h):
          q_table[((x1, y1), (x2, y2))] = [np.random.uniform(-5,0) for _ in range (4)]

  return q_table

def update_q_table(env, state, action, new_state, reward):
    max_future_q = np.max(q_table[new_state])
    q_table[state][action] += ALPHA*(reward + GAMMA * max_future_q - q_table[state][action])   

def epsilon_greedy_policy(epsilon, state):
  
    if random.uniform(0,1) < epsilon:
      return np.random.randint(0, env.action_space.n)
    else:
      return np.argmax(q_table[state])


def q_learning(env):
    epsilon = EPSILON
    scores = []
    wins = []

    for i in range(EPISODES):
        obs = env.reset()
        done = False
        episode_reward = 0

        if i % SHOW_EVERY == 0:
          print(f"on # {i}, epsilon: {epsilon}")
          print(f"{SHOW_EVERY} ep mean {np.mean(scores[-SHOW_EVERY:])}")
          show = True
        else:
          show = False

        while done == False:
          
            action = epsilon_greedy_policy(epsilon, obs)
            new_obs, reward, done, _ = env.step(action)
            episode_reward += reward

            if show:
              env.render()

            # if not done:
            update_q_table(env, obs, action, new_obs, reward) 
            
            if done:
              scores.append(episode_reward)
              wins.append(env.won)

            obs = new_obs       
          
        epsilon *= EPS_DECAY

    moving_avg = np.convolve(scores, np.ones((SHOW_EVERY,)) / SHOW_EVERY, mode='valid')
    plot_rewards(moving_avg, 'q_learning_rewards')

    victories_avg = np.convolve(wins, np.ones((SHOW_EVERY,)) / SHOW_EVERY, mode='valid')
    plot_victories(victories_avg, 'q_learning_victories_iterations')

    with open(NAME_TABLE + ".pickle", "wb") as f:
      pickle.dump(q_table, f)

    env.close()

if start_q_table is None:
  q_table = initialize_q_table(env)

else:
  print("LOADING Q-TABLE")
  with open(start_q_table, "rb") as f:
    q_table = pickle.load(f)

q_learning(env)