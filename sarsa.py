import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
import time

style.use("ggplot")

SIZE = 10

HM_EPISODES = 250000
MOVE_PENALTY = 1
ENEMY_PENALTY = 300
FOOD_REWARD = 25
CONST_EPSILON = 100
GAMA = 0.9

SHOW_EVERY = 3000  # how often to play through env visually.

# Em duvida se será que Q table ou V table...

start_q_table = None # None or Filename
start_nsa_table = None # None or Filename

PLAYER_N = 1  # player key in dict
FOOD_N = 2  # food key in dict
ENEMY_N = 3  # enemy key in dict

# the dict!
d = {1: (255, 175, 0),
     2: (0, 255, 0),
     3: (0, 0, 255)}


class Blob:
    def __init__(self):
        self.x = np.random.randint(0, SIZE)
        self.y = np.random.randint(0, SIZE)

    def __str__(self):
        return f"{self.x}, {self.y}"

    def __sub__(self, other):
        return (self.x-other.x, self.y-other.y)

    def action(self, choice):
        '''
        Gives us 4 total movement options. (0,1,2,3)
        '''
        if choice == 0:
            self.move(x=1, y=1)
        elif choice == 1:
            self.move(x=-1, y=-1)
        elif choice == 2:
            self.move(x=-1, y=1)
        elif choice == 3:
            self.move(x=1, y=-1)

    def move(self, x=False, y=False):

        # If no value for x, move randomly
        if not x:
            self.x += np.random.randint(-1, 2)
        else:
            self.x += x

        # If no value for y, move randomly
        if not y:
            self.y += np.random.randint(-1, 2)
        else:
            self.y += y


        # If we are out of bounds, fix!
        if self.x < 0:
            self.x = 0
        elif self.x > SIZE-1:
            self.x = SIZE-1
        if self.y < 0:
            self.y = 0
        elif self.y > SIZE-1:
            self.y = SIZE-1

def loadTable(states, places):
    """
    Receives states as array like and the quantity of actions in each state
    """
    table = {}
    for state in states:
        table[state] = np.zeros(places)
    return table


# Create a list to represent the quantity of states
states = []
for x1 in range(-SIZE+1, SIZE):
    for y1 in range(-SIZE+1, SIZE):
        for x2 in range(-SIZE+1, SIZE):
            for y2 in range(-SIZE+1, SIZE):
                states.append(((x1, y1),(x2, y2)))


if start_q_table is None:
    # initialize the q-table#
    q_table = loadTable(states, 4)

else:
    with open(start_q_table, "rb") as f:
        q_table = pickle.load(f)

if start_nsa_table is None:
    nsa_table = {}
else:
    with open(start_nsa_table, "rb") as f:
        nsa_table = pickle.load(f)

episode_rewards = []
optimalvalues = []

# Eligibility table
E_table = loadTable(states, 4)

#Loop through all the steps on the episode
rng = np.random.default_rng()
for episode in range(HM_EPISODES):

    player = Blob()
    food = Blob()
    enemy = Blob()

    for key in E_table:
        E_table[key] = np.zeros(4)

    expected_return = []
    actions = []
    ep_states = []

    if episode % SHOW_EVERY == 0:
        print(f"on #{episode}, mean epsilon is WRITE SOMETHING HERE")
        print(f"{SHOW_EVERY} TO BE DECEIDED")
        show = True
    else:
        show = False

    for step in range(200):
        obs = (player-food, player-enemy)
        ep_states.append(obs)

        # Chosing the next step

        maxq = np.max(q_table[obs])
        optimal_action = np.where(q_table[obs] == maxq)[0]
        if len(optimal_action) > 1:
            action = rng.choice(4)
        else:
            epsilon = CONST_EPSILON/(CONST_EPSILON + ns_table[obs])
            prob_actions = np.zeros(4) + epsilon/4
            prob_actions[optimal_action] += 1 - epsilon
            action = rng.choice(4, p = prob_actions)

        actions.append(action)
        player.action(action)

        if player.x == enemy.x and player.y == enemy.y:
            reward = -ENEMY_PENALTY
        elif player.x == food.x and player.y == food.y:
            reward = FOOD_REWARD
        else:
            reward = -MOVE_PENALTY

        expected_return.append(0)
        exp = len(expected_return)-1
        expected_return = [(lambda i, x: x + reward*(GAMA**(exp-i)))(i, x)
        for i, x in enumerate(expected_return)]

        if show:
            env = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)  # starts an rbg of our size
            env[food.x][food.y] = d[FOOD_N]  # sets the food location tile to green color
            env[player.x][player.y] = d[PLAYER_N]  # sets the player tile to blue
            env[enemy.x][enemy.y] = d[ENEMY_N]  # sets the enemy location to red
            img = Image.fromarray(env, 'RGB')  # reading to rgb. Apparently. Even tho color definitions are bgr. ???
            img = img.resize((300, 300))  # resizing so we can see our agent in all its glory.
            cv2.imshow("image", np.array(img))  # show it!
            if reward == FOOD_REWARD or reward == -ENEMY_PENALTY:  # crummy code to hang at the end if we reach abrupt end for good reasons or not.
                if cv2.waitKey(500) & 0xFF == ord('q'):
                    break
            else:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        if reward == FOOD_REWARD or reward == -ENEMY_PENALTY:
            break

    obs = (player-food, player-enemy)
    ep_states.append(obs)
    optimalvalue = 0

    for id, state in enumerate(ep_states):

        try:
            ns_table[state] += 1
        except:
            ns_table[state] = 1
        if id < len(actions):
            action =  actions[id]
            try:
                nsa_table[state][action] += 1
            except:
                nsa_table[state] = np.zeros(4)
                nsa_table[state][action] = 1
            alpha = 1/nsa_table[state][action]
            q_table[state][action] += alpha*(expected_return[id] - q_table[state][action])
            optimalvalue += q_table[state][action]
    optimalvalues.append(optimalvalue)

moving_avg = np.convolve(optimalvalues, np.ones((SHOW_EVERY,))/SHOW_EVERY, mode='valid')

plt.plot([i for i in range(len(moving_avg))], moving_avg)
plt.ylabel(f"Reward {SHOW_EVERY}ma")
plt.xlabel("episode #")
plt.show()
