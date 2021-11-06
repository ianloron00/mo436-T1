
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style

from tqdm import tqdm
from sarsa import MOVEMENT, Blob, SarsaLambda

style.use("ggplot")

MOVE_REWARD = -1
ENEMY_REWARD = -300
FOOD_REWARD = 100

SIZE = 10
N_EPISODES = 150_000
N_MAX_STEPS_IN_EPISODE = 10_000


_lambda = 0.4
_gamma = 0.99
_alpha = 0.1

movements = (MOVEMENT.UP, MOVEMENT.DOWN, MOVEMENT.LEFT, MOVEMENT.RIGHT)
movements_values = (MOVEMENT.UP.value, MOVEMENT.DOWN.value,
                    MOVEMENT.LEFT.value, MOVEMENT.RIGHT.value)

is_stochastic = True
for _lambda in [0, 0.2, 0.4, 0.6, 0.8, 1]:
    print(f'Lambda: {_lambda}')

    sarsa_learning = SarsaLambda(
        _gamma, _alpha, _lambda, movements_values)

    optimalvalues = []

    food = Blob(SIZE, SIZE - 1, SIZE - 1)

    for episode in tqdm(range(N_EPISODES)):

        player = Blob(SIZE, 0, 0)
        enemy = Blob(SIZE, 5, 5)

        steps = 0
        total_reward = 0

        sarsa_learning.new_episode()

        for step in range(N_MAX_STEPS_IN_EPISODE):

            if player == enemy or player == food:
                break

            state_before = (player-food, player-enemy)
            action = sarsa_learning.next_action(state_before)

            player.action(MOVEMENT(action))

            if is_stochastic:
                enemy_action = np.random.choice(movements)
                enemy.action(enemy_action)

            reward = 0
            if player == enemy:
                reward = ENEMY_REWARD
            elif player == food:
                reward = FOOD_REWARD
            else:
                reward = MOVE_REWARD
            total_reward += reward

            state_after = (player-food, player-enemy)
            sarsa_learning.update(state_before, action, reward, state_after)

        optimalvalues.append(total_reward)

    sarsa_learning.save_pickle()

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot([i for i in range(len(optimalvalues))], optimalvalues)
    ax.set_ylabel(f"Rewards")
    ax.set_xlabel("Episodes")
    ax.set_title(f'{sarsa_learning} - Stochastic')
    fig.savefig(f'sarsa/images/{repr(sarsa_learning)}.png')
# plt.show()
