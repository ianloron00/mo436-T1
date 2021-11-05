from dependencies import *
import os

def plot_rewards(moving_avg, title='rewards', save=True):
  plt.plot([i for i in range(len(moving_avg))], moving_avg)
  plt.ylabel(f"reward")
  plt.xlabel('episode #')
  if not os.path.isdir('./monte_carlo/images/'):
      os.makedirs('./monte_carlo/images/')
  if save:
    plt.savefig('monte_carlo/images/'+ title +'.png')
  plt.show()

def plot_victories(victories_avg, title='victories_iterations', save=True):
  plt.plot([i for i in range(len(victories_avg))], victories_avg)
  plt.title("victories per number iterations")
  if save:
    plt.savefig('monte_carlo/images/' + title + '.png')
  plt.show()
