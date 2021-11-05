import pickle

import numpy as np


class Sarsa():
    def __init__(self, _gamma: float, _lambda: float, _epsilon: float, q_table: dict, nsa_table: dict = {}, eligibility_traces: dict = {}, ns_table: dict = {}):
        self.gamma = _gamma
        self.llambda = _lambda
        self.epsilon = _epsilon
        self.q_table = q_table
        self.nsa_table = nsa_table
        self.ns_table = ns_table
        self.eligibility_traces_initialize = eligibility_traces
        self.rng = np.random.default_rng()

        self.episode_actions = None
        self.episode_states = None
        self.episode_reward = None
        self.alpha = None
        self.delta = None
        self.eligibility_traces = None

    def __str__(self) -> str:
        return f'Sarsa: $\gamma$ = {self.gamma} | $\lambda$ = {self.llambda} | $\epsilon$ = {self.epsilon}'

    def new_episode(self):
        self.episode_actions = []  # episode actions
        self.episode_states = []  # episode states
        self.episode_reward = []  # episode rewards

        self.eligibility_traces = self.eligibility_traces_initialize.copy()

    def update_episode(self, state, reward, action):
        self.episode_states.append(state)
        self.episode_reward.append(reward)
        self.episode_actions.append(action)

        self.nsa_table[state][action] += 1
        self.ns_table[state] += 1

    def get_total_reward(self):
        return np.sum(self.episode_reward)

    def update_tables(self):
        delta = self.episode_reward[-1] + self.gamma*self.q_table[self.episode_states[-1]
                                                                  ][self.episode_actions[-1]] - self.q_table[self.episode_states[-2]][self.episode_actions[-2]]
        self.eligibility_traces[self.episode_states[-2]] += 1

        for step, state in enumerate(self.episode_states):
            alpha = 1/(self.nsa_table[state]
                       [self.episode_actions[step]])

            self.q_table[self.episode_states[-2]][self.episode_actions[-2]
                                                  ] += alpha * delta * self.eligibility_traces[state][self.episode_actions[step]]

            self.eligibility_traces[state][self.episode_actions[step]
                                           ] *= self.gamma * self.llambda

    def epsilon_greedy_policy(self, state):
        max_value = np.max(self.q_table[state])
        opt_action = np.where(self.q_table[state] == max_value)[0]

        # action = np.random.choice(opt_action)
        if len(opt_action) > 1:
            action = np.random.choice(opt_action)
        else:
            epsilon = self.epsilon/(self.epsilon + self.ns_table[state])
            prob_actions = np.zeros(4) + epsilon/4
            prob_actions[opt_action] += 1 - epsilon
            action = self.rng.choice(4, p=prob_actions)

        return action
