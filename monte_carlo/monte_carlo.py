import numpy as np
import random
from q_learning.basis.dependencies import *

class MonteCarlo():
    def __init__(self, gamma=0.90, const_epsilon=100):

        self.const_epsilon = const_epsilon
        self.gamma = gamma
        self.rng = np.random.default_rng()
        self.states = []
        self.actions = []
        self.expected_return = []

    def initialize (self, env):

        #if name_table is None:
        print("CREATING Q-TABLE, NS-TABLE and NSA-TABLE")
        self.q_table = {}
        self.ns_table = {}
        self.nsa_table = {}
        h = env.height
        w = env.width
        for x1 in range(-w+1, w):
            for y1 in range(-h+1, h):
                for x2 in range(-w+1, w):
                    for y2 in range(-h+1, h):
                        self.q_table[((x1, y1), (x2, y2))] = np.zeros(4)
        # else:
        #     print("LOADING Q-TABLE")
        #     with open("q_learning/q_tables/" + name_table, "rb") as f:
        #         self.q_table = pickle.load(f)

        # return self.q_table

    def update (self, env, state, action, new_state, reward, done):

        self.states.append(state)
        self.actions.append(action)
        self.expected_return.append(0)
        exp = len(self.expected_return)-1
        self.expected_return = [(
        lambda i, x: x + reward*(self.gamma**(exp-i)))(i, x)
        for i, x in enumerate(self.expected_return)]

        if done:

            for id, oldstate in enumerate(self.states):
                try:
                    self.ns_table[oldstate] += 1
                except:
                    self.ns_table[oldstate] = 1
                action = self.actions[id]
                try:
                    self.nsa_table[oldstate][self.actions[id]] += 1
                except:
                    self.nsa_table[oldstate] = np.zeros(4)
                    self.nsa_table[oldstate][self.actions[id]] = 1
                alpha = 1/(self.nsa_table[oldstate][self.actions[id]])
                self.q_table[oldstate][self.actions[id]] += alpha*(self.expected_return[id] - self.q_table[oldstate][self.actions[id]])
            self.expected_return = []
            self.actions = []
            self.states = []

    def count_states(self):
        return len(self.ns_table.keys())


    def epsilon_greedy_policy(self, env, state):

        maxq = np.max(self.q_table[state])
        opt_action = np.where(self.q_table[state] == maxq)[0]
        if len(opt_action) > 1:
            action = self.rng.choice(4)

        else:
            epsilon = self.const_epsilon/(self.const_epsilon + self.ns_table[state])
            prob_actions = np.zeros(4) + epsilon/4
            prob_actions[opt_action] += 1 - epsilon
            action = self.rng.choice(4, p = prob_actions)
        return action

    def greedy_policy(self, env, state):

        maxq = np.max(self.q_table[state])
        opt_action = np.where(self.q_table[state] == maxq)[0]
        return self.rng.choice(opt_action)
