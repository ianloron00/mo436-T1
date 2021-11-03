import os
from datetime import datetime
import numpy as np
from monte_carlo.basis.environment import *
from monte_carlo.basis.graphs import *
# from graphs import plot_rewards, plot_victories
import pickle
from monte_carlo.basis.dependencies import *
from monte_carlo.monte_carlo import MonteCarlo

isTraining = True
isStochastic = True
SAVE_IMAGES = True
PATH = '.data'

EPISODES = 2_000_000 if isTraining else 10
GAMMA = 0.90 if isTraining else 0.0
CONST_EPSILON = 1000 if isTraining else 0

SHOW_EVERY = int(EPISODES/200) if isTraining else 1

NOW = datetime.now().strftime('%Y%m%d_%H%M%S')
NAME_COMPLEMENT = '_stochastic' if isStochastic else '_deterministic'
NAME_Q_TABLE = 'monte_carlo_Q_table' + NAME_COMPLEMENT + NOW
NAME_SCORES = 'monte_carlo_Scores' + NAME_COMPLEMENT + NOW
NAME_NSA_TABLE = 'monte_carlo_NSA_table' + NAME_COMPLEMENT + NOW
SAVE_TABLE = True if isTraining else False

# NOTE: In case of a normal run of the algorithm it will be necessary to insert the files below

if not os.path.isdir('./data'):
    os.makedirs('./data')

start_q_table = None # None or Filename
start_ns_table = None # None or Filename
start_nsa_table = None # None or Filename

start_q_table = None if isTraining else NAME_Q_TABLE + '.pickle'
start_ns_table = None if isTraining else NAME_NS_TABLE + '.pickle'
start_nsa_table = None if isTraining else NAME_NSA_TABLE + '.pickle'

def monte_carlo(env):
    agent = MonteCarlo(gamma=GAMMA)
    agent.initialize(env)


    scores = []
    wins = []
    time_fast = 5 if isTraining else 30
    time_slow = 400

    for i in range(EPISODES):
        obs = env.reset()
        done = False
        episode_reward = 0

        if i % SHOW_EVERY == 0:
            print(f"on # {i}, the known states are {agent.count_states()}")
            if i != 0:
                print(f"{SHOW_EVERY} ep mean {np.mean(scores[-SHOW_EVERY:])}")
            show = False
        else:
            show = False

        while done == False:
            if isTraining:
                action = agent.epsilon_greedy_policy(env, obs)
            else:

                action = agent.greedy_policy(env, obs)
            new_obs, reward, done, _ = env.step(action)
            episode_reward += reward

            if show:
                env.render(time_fast=time_fast, time_slow=time_slow)

            agent.update(env, obs, action, new_obs, reward, done)

            if done:
                scores.append(episode_reward)
                wins.append(env.won)

            obs = new_obs

    if SAVE_IMAGES:
        print("SAVING IMAGES")

    reward_title = 'monte_carlo_rewards_10x10' + NAME_COMPLEMENT
    if not isTraining:
        reward_title += '_after_training'
    moving_avg = np.convolve(scores, np.ones((SHOW_EVERY,)) / SHOW_EVERY, mode='valid')
    plot_rewards(moving_avg, title=reward_title, save=SAVE_IMAGES)

    victories_title = 'monte_carlo_victories_10x10' + NAME_COMPLEMENT
    if not isTraining:
        victories_title += '_after_training'
    victories_avg = np.convolve(wins, np.ones((SHOW_EVERY,)) / SHOW_EVERY, mode='valid')
    plot_victories(victories_avg, title=victories_title, save=SAVE_IMAGES)

    if (SAVE_TABLE):
        path = 'monte_carlo/q_tables/' + NAME_Q_TABLE + '.pickle'
        path2 = 'monte_carlo/scores/' + NAME_SCORES + '.pickle'
        print(f"SAVING {NAME_Q_TABLE}")
        with open(path, "wb") as f:
          pickle.dump(agent.q_table, f)
        with open(path2, "wb") as f:
          pickle.dump(scores, f)

    env.close()

def main():
    env = GameEnv(stochastic=isStochastic)
    monte_carlo(env)

if __name__ == '__main__':
    main()
