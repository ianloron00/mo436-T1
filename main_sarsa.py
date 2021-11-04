import pickle

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style
from PIL import Image

from tqdm import tqdm
from sarsa import MOVEMENT, REWARD, Blob, Sarsa, Tables

style.use("ggplot")

MOVE_REWARD = -1
ENEMY_REWARD = -300
FOOD_REWARD = 25

SIZE = 10
N_EPISODES = 1000
N_MAX_STEPS_IN_EPISODE = 100

CONST_EPSILON = 100
GAMA = 0.9
LAMBDA = 0.2

SHOW_EVERY = 3000  # how often to play through env visually.

# Em duvida se será que Q table ou V table...

PLAYER_N = 1  # player key in dict
FOOD_N = 2  # food key in dict
ENEMY_N = 3  # enemy key in dict

q_table = Tables.new_table(SIZE, 4)
nsa_table = Tables.new_table(SIZE, 4)  # quantidade de ações por estado
eligibility_traces = Tables.new_table(SIZE, 4)

sarsa = Sarsa(0.99, 0.1, 0.1, q_table, nsa_table, eligibility_traces)

optimalvalues = []
for episode in tqdm(range(N_EPISODES)):

    player = Blob(SIZE)
    food = Blob(SIZE)
    enemy = Blob(SIZE)

    sarsa.new_episode()

    state = (player-food, player-enemy)
    sarsa.append_episode_state(state)
    action = sarsa.get_action(state)
    sarsa.set_nsa_table(state, action)

    # if episode % SHOW_EVERY == 0:
    #     print(f"on #{episode}, mean epsilon is WRITE SOMETHING HERE")
    #     print(f"{SHOW_EVERY} TO BE DECEIDED")
    #     show = False
    # else:
    #     show = False

    for step in range(N_MAX_STEPS_IN_EPISODE):

        player.action(MOVEMENT(action))

        reward = 0
        if player == enemy:
            reward = ENEMY_REWARD
        elif player == food:
            reward = FOOD_REWARD
        else:
            reward = MOVE_REWARD
        sarsa.set_reward(reward)

        state = (player-food, player-enemy)
        sarsa.append_episode_state(state)

        action = sarsa.get_action(state)
        sarsa.set_nsa_table(state, action)

        sarsa.update()

        # if False:
        #     # starts an rbg of our size
        #     env = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
        #     # sets the food location tile to green color
        #     env[food.x][food.y] = d[FOOD_N]
        #     # sets the player tile to blue
        #     env[player.x][player.y] = d[PLAYER_N]
        #     # sets the enemy location to red
        #     env[enemy.x][enemy.y] = d[ENEMY_N]
        #     # reading to rgb. Apparently. Even tho color definitions are bgr. ???
        #     img = Image.fromarray(env, 'RGB')
        #     # resizing so we can see our agent in all its glory.
        #     img = img.resize((300, 300))
        #     cv2.imshow("image", np.array(img))  # show it!
        #     # crummy code to hang at the end if we reach abrupt end for good reasons or not.
        #     if player == enemy or player == food:
        #         if cv2.waitKey(500) & 0xFF == ord('q'):
        #             break
        #     else:
        #         if cv2.waitKey(1) & 0xFF == ord('q'):
        #             break

        # if player == enemy or player == food:
        #     break

    optimalvalues.append(sarsa.episode_reward)


moving_avg = np.convolve(optimalvalues, np.ones(
    (SHOW_EVERY,))/SHOW_EVERY, mode='valid')

plt.plot([i for i in range(len(moving_avg))], moving_avg)
plt.ylabel(f"Reward {SHOW_EVERY}ma")
plt.xlabel("episode #")
plt.show()
