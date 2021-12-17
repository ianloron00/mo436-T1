import gym
from numpy.lib.utils import info
from reinforce.basis.dependencies import *
from reinforce.basis.agent  import Agent
from reinforce.basis.player import Player
from reinforce.basis.hunter import Hunter
from reinforce.basis.target import Target
from reinforce.basis.graphs import *
from reinforce.basis.game_display import *
import gym
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete

"""
should contain:
init
make
reset
render
step
close
(validate_action)
"""

SIZE = 10
MOVE_PENALTY = 1
HUNTER_PENALTY = 3000
TARGET_REWARD = 100
COLLISION_PENALTY = 10

class GameEnv:
    # change it not to overlay
    def __init__(self, width=SIZE, height=SIZE, stochastic=True, tabular=True):
        self.width = width
        self.height = height
        # reward in one timestep
        self.reward = 0
        # cumulative reward
        self.cum_reward = 0

        self.player = None
        self.hunter = None
        self.target = None

        self.isStochastic = stochastic
        self.isTabular = tabular

        if tabular:
            # ((player - target),(player - enemy))
            self.observation_space = ((-1,-1),(-1,-1))
        else: 
            # will contain multiple parameters.
            self.observation_space = np.zeros(shape=(3,8))
        # from 0 to 3.
        self.action_space = Discrete(4,)

        self.done = False
        self.info = {}

        self.timestep = 0
        self.reward = 0

        self.won = False

        self.display = Display(width, height)

    def make(env="myEnv"):
        return

    def reset(self):
        self.done = False
        self.reward = 0
        self.cum_reward = 0
        self.timestep = 0 
        self.won = False

        pX, pY, hX, hY, tX, tY = self.get_positions()

        self.player = Player(pX,pY)
        self.hunter = Hunter(hX, hY)
        self.target = Target(tX, tY)

        if not self.isTabular:
            self.observation_space = np.zeros(shape=(3,8))
        self.update_observation_space()
        return self.observation_space

    def step(self, action):
        # self.target.move(self)
        if self.isStochastic:
            self.hunter.move(self)

        moved = self.player.action(self, action)

        self.update_observation_space()
        
        self.update_reward(moved=moved)
        self.cum_reward += self.reward

        return self.observation_space, self.reward, self.done, self.info

    def render(self, time_fast=10, time_slow=500):
        # redefine
        self.display.render(self, time_slow=time_slow, time_fast=time_fast)

    def close(self):
        self.display.quit()

    def update_observation_space(self):
        if self.isTabular:
            self.observation_space = (self.player - self.target, self.player - self.hunter)
       
        else:
            new_obs = np.array([self.width, self.height, self.player.x, self.player.y, 
                                self.hunter.x, self.hunter.y, self.target.x, self.target.y])
            
            # in the first line there will be the newest values.
            self.observation_space = np.vstack((new_obs, self.observation_space[:-1]))

    def update_reward(self, moved=True):
        if self.player == self.hunter:
            self.reward = -HUNTER_PENALTY
            self.done = True
        
        elif self.player == self.target:
            self.reward = TARGET_REWARD
            self.won = True
            self.done = True
        
        else:
            self.reward = -MOVE_PENALTY
        
        if not moved:
            self.reward -= COLLISION_PENALTY

    def get_positions(self):
        w, h = self.width - 1, self.height - 1

        pX, pY, hX, hY, tX, tY = 0, 0, int(w/2), int(h/2), w, h

        if self.isStochastic:
            tX = np.random.randint(0, self.width)
            tY = np.random.randint(0, self.height)
            pX, pY, hX, hY = tX, tY, tX, tY

            while(abs(pX - tX) <= 1 and (pY - tY) <= 1):
                pX = np.random.randint(0, self.width)
                pY = np.random.randint(0, self.height)

            while(abs(hX - tX) <= 1 and (hY - tY) <= 1 or hX == pX and hY == pY):
                hX = np.random.randint(0, self.width)
                hY = np.random.randint(0, self.height)

        return pX, pY, hX, hY, tX, tY