import time

from sarsa.tables import BaseTable
from . import QTable
from . import BaseTable
import time
import pickle


class SarsaLambda:
    def __init__(self, gamma, alpha, _lambda, actions):
        self.gamma = gamma
        self.alpha = alpha
        self._lambda = _lambda
        self.actions = actions
        self.eligibility_traces = None
        self.q_values = QTable(actions)
        self.episode = 0
        self.episode_reward = 0

        self.timestamp = str(time.time()).replace(".", "_")

    def new_episode(self):
        self.eligibility_traces = BaseTable()
        self.episode += 1
        self.episode_reward = 0

    def next_action(self, state):
        return self.q_values.get_greedy_action(state)

    def update(self, state_before, action, reward, state_after):
        expected_reward = self.q_values.get_expected_reward(
            state_before, action)
        next_action = self.q_values.get_greedy_action(state_after)
        next_expected_reward = self.q_values.get_expected_reward(
            state_after, next_action)

        delta = reward + self.gamma * next_expected_reward - expected_reward

        self.eligibility_traces.increment(state_before, action)
        self.q_values.ensure_exists(state_before, action)

        def update_q_values(state, action):
            old_expected_reward = self.q_values.get_expected_reward(
                state, action)
            updated_expected_reward = old_expected_reward + self.alpha * \
                delta * self.eligibility_traces.get(state, action)

            self.q_values.set_expected_reward(
                state, action, updated_expected_reward)
            self.eligibility_traces.decay(
                state, action, self.gamma * self._lambda)

        self.q_values.for_each(update_q_values)
        self.episode_reward += reward

    def dump(self):
        return self.q_values.get_all_values(),

    def load(self, values):
        self.q_values.set_all_values(values)

    def load_pickle(self, path):
        with open(path, 'rb') as pkl_file:
            self.load(pickle.load(pkl_file))

    def save_pickle(self):
        with open(f'sarsa/q_table/sarsa_lambda_{self.timestamp}.pickle', 'wb') as pkl_file:
            pickle.dump(self.dump(), pkl_file,
                        protocol=pickle.HIGHEST_PROTOCOL)

    def __str__(self) -> str:
        return f'Sarsa Lambda: $\gamma$ = {self.gamma} | $\lambda$ = {self._lambda}'

    def __repr__(self) -> str:
        return f'sarsa_lambda_{self.timestamp}'
