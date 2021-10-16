from dependencies import *

class agent:
  def __init__(self, x=np.random.randint(0,10), y=np.random.randint(0,10)):
    self.x = x
    self.y = y
  
  def __sub__(self, other):
    return (self.x - other.x, self.y - other.y)
  
  def __eq__(self, o: object) -> bool:
      return (self.x == o.x and self.y == o.y)

  def __str__(self):
    return f"{self.x}, {self.y}"
  
  def choose_action(self, env):
    action = env.action_space.sample()
    return action
  
  def set_position(self, x, y):
    self.x = x
    self.y = y
  
  def get_position(self):
    return (self.x, self.y)

  def action(self, env, choice):
    if choice == 0:
      self.move(env, x=1, y=0)
    elif choice == 1:
      self.move(env, x=-1, y=0)
    elif choice == 2:
      self.move(env, x=0, y=1)
    elif choice == 3:
      self.move(env, x=0, y=-1)
    # if choice == 0:
      # self.move(env, x=1, y=1)
    # elif choice == 1:
      # self.move(env, x=-1, y=-1)
    # elif choice == 2:
      # self.move(env, x=-1, y=1)
    # elif choice == 3:
      # self.move(env, x=1, y=-1)

  def move(self, env, x=False, y=False):
    if not x:
      self.x += np.random.randint(-1, 2)
    else:
       self.x += x
    
    if not y:
      self.y += np.random.randint(-1, 2)
    else:
       self.y += y

    if self.x < 0:
      self.x = 0
    elif self.x > env.width-1:
      self.x = env.width-1
    
    if self.y < 0:
      self.y = 0
    elif self.y > env.height-1:
      self.y = env.height-1