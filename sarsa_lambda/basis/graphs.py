from dependencies import *
import os

def plot_rewards(moving_avg, title='rewards',lambd = 0, N0 = 0, save=True):
  plt.plot([i for i in range(len(moving_avg))], moving_avg)
  plt.ylabel(f"reward")
  plt.xlabel('episode #')
  plt.title(r'$\lambda = {}; N0 = {}$'.format(lambd, N0))
  if not os.path.isdir('./monte_carlo/images/'):
      os.makedirs('./monte_carlo/images/')
  if save:
    plt.savefig('sarsa_lambda/images/'+ title + '_lambda_' + str(lambd) + '_NO_' + str(N0) +'.png')
  plt.close()


def plot_victories(victories_avg, title='victories_iterations',lambd = 0, N0 = 0, save=True):
  plt.plot([i for i in range(len(victories_avg))], victories_avg)
  plt.title(r'victories per number iterations $\lambda = {}; N0 = {}$'.format(lambd, N0))
  if save:
    plt.savefig('sarsa_lambda/images/' + title + '_lambda_' + str(lambd) + '_NO_' + str(N0) +'.png')
  plt.close()
