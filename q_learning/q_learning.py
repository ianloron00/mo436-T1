import numpy as np
import random
from q_learning.basis.dependencies import *

class QLearning():
    def __init__(self, alpha=0.1, gamma=0.95, epsilon=0.99, n_episodes=100_000, 
                isTraining=False, name_table='q_learning_table'):

        self.ALPHA = alpha
        self.GAMMA = gamma
        # self.isTraining = isTraining

    # self.q_table shape: [((+-height, +-width), (+-height, +-width))][4]
    def initialize (self, env, name_table='q_learning_table.pickle'):
        
        if name_table is None:
            print("CREATING Q-TABLE")
            self.q_table = {}
            h = env.height    
            w = env.width
            for x1 in range(-w+1, w):
                for y1 in range(-h+1, h):
                    for x2 in range(-w+1, w):
                        for y2 in range(-h+1, h):
                            self.q_table[((x1, y1), (x2, y2))] = [np.random.uniform(-5,0) for _ in range (4)]
        else:
            print("LOADING Q-TABLE")
            with open("q_learning/q_tables/" + name_table, "rb") as f:
                self.q_table = pickle.load(f)

        # return self.q_table

    def update (self, env, state, action, new_state, reward):

        max_future_q = np.max(self.q_table[new_state])
        self.q_table[state][action] += self.ALPHA*(reward + self.GAMMA * max_future_q - self.q_table[state][action])   

    def epsilon_greedy_policy(self, env, epsilon, state):
    
        if random.uniform(0,1) < epsilon:
            return np.random.randint(0, env.action_space.n)
        else:
            return np.argmax(self.q_table[state])
