import pickle

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style
from PIL import Image

from tqdm import tqdm
from sarsa import MOVEMENT, REWARD, Blob, Sarsa, Tables

style.use("ggplot")


def moving_average(a, n=10):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


MOVE_REWARD = -1
ENEMY_REWARD = -300
FOOD_REWARD = 100

SIZE = 10
N_EPISODES = 50_000
N_MAX_STEPS_IN_EPISODE = 200

q_table = Tables.new_table(SIZE, 4)
nsa_table = Tables.new_table(SIZE, 4)
eligibility_traces = Tables.new_table(SIZE, 4)
ns_table = Tables.new_table(SIZE, 1)

for llambda in np.array([0, 0.2, 0.4, 0.6, 0.8, 1]):
    print(f'Lambda = {llambda}')
    sarsa_learning = Sarsa(0.95, llambda, 0.85, q_table, nsa_table,
                           eligibility_traces, ns_table)

    optimalvalues = []
    for episode in tqdm(range(N_EPISODES)):

        player = Blob(SIZE)
        food = Blob(SIZE)
        enemy = Blob(SIZE)

        episode_rewards = 0

        sarsa_learning.new_episode()

        for step in range(N_MAX_STEPS_IN_EPISODE):

            if player == enemy or player == food:
                break

            episode_state = (player-food, player-enemy)
            action = sarsa_learning.epsilon_greedy_policy(episode_state)

            player.action(MOVEMENT(action))

            reward = 0
            if player == enemy:
                reward = ENEMY_REWARD
            elif player == food:
                reward = FOOD_REWARD
            else:
                reward = MOVE_REWARD

            episode_rewards += reward
            sarsa_learning.update_episode(episode_state, reward, action)

            if step == 0:
                continue

            sarsa_learning.update_tables()

        optimalvalues.append(episode_rewards)

    moving_avg = moving_average(optimalvalues)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot([i for i in range(len(optimalvalues))], optimalvalues)
    ax.set_ylabel(f"Rewards")
    ax.set_xlabel("Episodes")
    ax.set_title(f'{sarsa_learning}')
    fig.savefig(f'sarsa/{llambda}.png')
    # plt.show()
