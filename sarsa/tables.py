import numpy as np
import pickle


class Tables:

    @staticmethod
    def new_table(n_size, n_actions):
        table = dict()
        size = range(-n_size + 1, n_size)
        for a in size:
            for b in size:
                for c in size:
                    for d in size:
                        table[((a, b), (c, d))] = np.zeros(n_actions)

        return table

    @staticmethod
    def load_table(path):
        table = dict()
        with open(path, "rb") as f:
            table = pickle.load(f)
        return table
