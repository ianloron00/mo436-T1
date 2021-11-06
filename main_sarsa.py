
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style

from tqdm import tqdm
from sarsa import MOVEMENT, Blob, SarsaLambda

style.use("ggplot")


def moving_average(a, n=10):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


MOVE_REWARD = -1
ENEMY_REWARD = -300
FOOD_REWARD = 100

SIZE = 10
N_EPISODES = 30_000
N_MAX_STEPS_IN_EPISODE = 1_000


_lambda = 0.4
_gamma = 0.99
_alpha = 0.1
_lambda = 0.1
_epsilon = 0.1

movements = (MOVEMENT.UP, MOVEMENT.DOWN, MOVEMENT.LEFT, MOVEMENT.RIGHT)
movements_values = (MOVEMENT.UP.value, MOVEMENT.DOWN.value,
                    MOVEMENT.LEFT.value, MOVEMENT.RIGHT.value)


# for _epsilon in 10**np.arange(-1, 3).astype(float):
for _lambda in np.linspace(0, 1, 6):

    sarsa_learning = SarsaLambda(
        _gamma, _alpha, _lambda, _epsilon, movements_values)

    optimalvalues = []

    food = Blob(SIZE, SIZE - 1, SIZE - 1)

    for episode in tqdm(range(N_EPISODES)):

        player = Blob(SIZE, 0, 0)
        enemy = Blob(SIZE)

        steps = 0
        total_reward = 0

        sarsa_learning.new_episode()

        for step in range(N_MAX_STEPS_IN_EPISODE):

            if player == enemy or player == food:
                break

            state_before = (player-food, player-enemy)
            action = sarsa_learning.next_action(state_before)

            player.action(MOVEMENT(action))
            reward = 0
            if player == enemy:
                reward = ENEMY_REWARD
            elif player == food:
                reward = FOOD_REWARD
            else:
                reward = MOVE_REWARD
            total_reward += reward

            state_after = (player-food, player-enemy)
            sarsa_learning.update(
                state_before, action, reward, state_after)

        optimalvalues.append(total_reward)

    sarsa_learning.save_pickle()
    moving_avg = moving_average(optimalvalues)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot([i for i in range(len(optimalvalues))], optimalvalues)
    ax.set_ylabel(f"Rewards")
    ax.set_xlabel("Episodes")
    ax.set_title(f'{sarsa_learning}')
    fig.savefig(f'sarsa/images/{repr(sarsa_learning)}.png')
# plt.show()
