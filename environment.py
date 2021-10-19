from pickle import FALSE
import gym
from numpy.lib.utils import info
from dependencies import *
from player import Player
from hunter import Hunter
from target import Target
from graphs import *
from game_display import *
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
HUNTER_PENALTY = 300
PORTAL_REWARD = 25

class GameEnv:
  # change it not to overlay
  def __init__(self, width=SIZE, height=SIZE):
    self.width = width
    self.height = height
    # reward in one timestep
    self.reward = 0
    # cumulative reward
    self.cum_reward = 0
    
    self.player = None
    self.hunter = None
    self.target = None

    # ((player - target),(player - enemy))
    self.observation_space = ((-1,-1),(-1,-1))
    # from 0 to 3.
    self.action_space = Discrete(4,)

    # it is not standard in gym, but rather useful for us.
    self.display_space = Box(shape=(height, width, 3), low=np.zeros(shape=(height, width, 3)), 
                             high=255*np.ones(shape=(height, width, 3)), dtype=np.uint8)

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

    # half_width = (self.width-1)/2
    # half_height = (self.height-1)/2

    targetX = np.random.randint(0, self.width)
    targetY = np.random.randint(0, self.height)
    pX, pY, hX, hY = targetX, targetY, targetX, targetY
    
    while(abs(pX - targetX) <= 1 and (pY - targetY) <= 1):
        pX = np.random.randint(0, self.width)
        pY = np.random.randint(0, self.height)

    while(abs(hX - targetX) <= 1 and (hY - targetY) <= 1 or hX == pX and hY == pY):
        hX = np.random.randint(0, self.width)
        hY = np.random.randint(0, self.height)


    self.player = Player(pX,pY)
    self.hunter = Hunter(hX, hY)
    self.target = Target(targetX, targetY)

    # self.hunter = Hunter(self.width -1 - x, y)
    # self.target = Target(self.width -1 -x , self.height -1 -y)

    self.update_observation_space()
    return self.observation_space

  def step(self, action):
    # self.target.move(self)
    self.hunter.move(self)
    self.player.action(self, action)

    self.update_observation_space()
    self.update_reward()
    self.cum_reward += self.reward

    return self.observation_space, self.reward, self.done, self.info
  
  def render(self, time_fast=10, time_slow=500):
    # redefine
    self.display.render(self, time_slow=time_slow, time_fast=time_fast)
  
  def close(self):
    self.display.quit()

  def update_observation_space(self):
    self.observation_space = (self.player - self.target, self.player - self.hunter)

  def update_reward(self):
    if self.player == self.hunter:
      self.reward = -HUNTER_PENALTY
      self.done = True
    elif self.player == self.target:
      self.reward = PORTAL_REWARD
      self.won = True
      self.done = True
    else:
      self.reward = -MOVE_PENALTY