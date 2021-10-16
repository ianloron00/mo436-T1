import gym
from numpy.lib.utils import info
from dependencies import *
from player import Player
from hunter import Hunter
from portal import Portal
from display import *
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
    self.reward = 0
    
    self.player = None
    self.hunter = None
    self.portal = None

    # ((player - portal),(player - enemy))
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
  
  def make(env="myEnv"):
    return

  def reset(self):
    self.done = False
    self.reward = 0
    self.timestep = 0 

    half_width = (self.width-1)/2
    half_height = (self.height-1)/2

    x = np.random.randint(0, half_width)
    y = np.random.randint(0, half_height)

    self.player = Player(x,y)
    self.hunter = Hunter(self.width -1 - x, y)
    self.portal = Portal(self.width -1 -x , self.height -1 -y)

    self.update_observation_space()

    self.won = False

    return self.observation_space

  def step(self, action):
    # self.portal.move(self)
    # self.hunter.move(self)
    self.player.action(self, action)

    self.update_reward()
    self.update_observation_space()
    
    return self.observation_space, self.reward, self.done, self.info
  
  def render(self):
    # redefine
    show_display(self)
  
  def close(self):
    return

  def update_observation_space(self):
    self.observation_space = (self.player - self.portal, self.player - self.hunter)

  def update_reward(self):
    if self.player == self.hunter:
      self.reward = -HUNTER_PENALTY
      self.done = True
    elif self.player == self.portal:
      self.reward = PORTAL_REWARD
      self.won = True
      self.done = True
    else:
      self.reward = -MOVE_PENALTY