import pickle
import numpy as np


class Sarsa():
    def __init__(self, _gamma: float, _lambda: float, _epsilon: float, q_table: dict, nsa_table: dict, eligibility_traces: dict):
        self.gamma = _gamma
        self.llambda = _lambda
        self.epsilon = _epsilon
        self.q_table = q_table
        self.nsa_table = nsa_table
        self.eligibility_traces_initialize = eligibility_traces

        self.player = None
        self.target = None
        self.enemy = None

        self.actions = None
        self.episode_states = None
        self.episode_reward = None
        self.alpha = None
        self.delta = None
        self.eligibility_traces = None

    def load_nsa_table(self, path):
        with open(path, "rb") as f:
            self.nsa_table = pickle.load(f)

    def new_episode(self):
        self.player = None
        self.target = None
        self.enemy = None

        self.actions = []
        self.episode_states = []
        self.episode_reward = []
        self.alpha = 0
        self.delta = 0
        self.eligibility_traces = self.eligibility_traces_initialize.copy()

    def append_episode_state(self, state):
        self.episode_states.append(state)

    def set_reward(self, reward):
        self.episode_reward.append(reward)

    def get_total_reward(self):
        return np.sum(self.episode_reward)

    def update(self):
        delta = self.episode_reward[-1] + self.gamma*self.q_table[self.episode_states[-1]
                                                                  ][self.actions[-1]] - self.q_table[self.episode_states[-2]][self.actions[-2]]
        self.eligibility_traces[self.episode_states[-2]] += 1

        for step, state in enumerate(self.episode_states):
            alpha = 1/self.nsa_table[state][self.actions[step]]
            self.q_table[self.episode_states[-2]][self.actions[-2]
                                                  ] += alpha * delta * self.eligibility_traces[state][self.actions[step]]

            self.eligibility_traces[state][self.actions[step]
                                           ] *= self.gamma * self.llambda

    def set_nsa_table(self, state, action: int):
        self.nsa_table[state][action] += 1

    def get_action(self, state):
        max_value = np.max(self.q_table[state])
        action = np.random.choice(
            np.where(self.q_table[state] == max_value)[0])
        self.actions.append(action)
        return action
