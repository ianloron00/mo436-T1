from . import BaseTable


class EligibilityTraces:
    def __init__(self, decay_rate=1):
        self.decay_rate = decay_rate
        self.values = BaseTable()

    def decay(self, state, action):
        self.values.update(state, action, lambda v: v * self.decay_rate)

    def increment(self, state, action):
        self.values.update(state, action, lambda v: v + 1, 1)

    def get(self, state, action):
        return self.values.get(state, action)
