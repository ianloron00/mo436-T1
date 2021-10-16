from dependencies import *

PLAYER_N = 1
PORTAL_N = 2
HUNTER_N = 3
WALL_N = 4

# GBR color
d = {1: (255, 175, 0),
     2: (0, 255, 0),
     3: (0, 0, 255),
     4: (255, 255, 255)}


def show_display(env, time_fast=1, time_slow=500):
  display = np.zeros((env.height, env.width, 3), dtype=np.uint8)
  display[env.portal.y][env.portal.x] = d[PORTAL_N]
  display[env.player.y][env.player.x] = d[PLAYER_N]
  display[env.hunter.y][env.hunter.x] = d[HUNTER_N]

  img = Image.fromarray(display, "RGB")
  k = env.height/env.width
  img = img.resize((1000, int(1000*k)))
  cv2.imshow("image", np.array(img))
  if env.done:
    if cv2.waitKey(time_slow) and 0xFF == ord("q"):
      return True
  else:
    if cv2.waitKey(time_fast) and 0xFF == ord("q"):
      return True

def plot_rewards(moving_avg, title='rewards'):
  plt.plot([i for i in range(len(moving_avg))], moving_avg)
  plt.ylabel(f"reward")
  plt.xlabel('episode #')
  plt.savefig('images/'+ title +'.png')
  plt.show()

def plot_victories(victories_avg, title='victories_iterations'):
  plt.plot([i for i in range(len(victories_avg))], victories_avg)    
  plt.title("victories per number iterations")
  plt.savefig('images/' + title + '.png')
  plt.show()