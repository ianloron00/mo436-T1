import numpy as np
from environment import * 
from graphs import *
# from graphs import plot_rewards, plot_victories
import pickle
from dependencies import *
from q_learning.q_learning import QLearning
from q_learning.q_extractor_features import *

isTraining = True
isStochastic = False
SAVE_IMAGES = False

EPISODES = 25_000 if isTraining else 10
GAMMA = 0.95 if isTraining else 0.0
EPSILON = 0.99 if isTraining else 0.005
# achieves half of epsilon at 1/K-th of the number of timesteps.
# decay = 2**(-log2(0.5/EPSILON) / (EPISODES / K)) ~ 2**(-1 / (EPISODES / K)) ~ 2**(-K / EPISODES)
EPS_DECAY = 2**(-4 / EPISODES) if isTraining else 1

SHOW_EVERY = int(EPISODES/10) if isTraining else 1

NAME_COMPLEMENT = '_stochastic' if isStochastic else '_deterministic'
NAME_TABLE = 'q_learning_10x10' + NAME_COMPLEMENT
SAVE_TABLE = True if isTraining else False


start_q_table = None if isTraining else NAME_TABLE + '.pickle'

def q_learning(env):
    q = QLearning(gamma=GAMMA)
    q.initialize(env, name_table=start_q_table)

    epsilon = EPSILON
    scores = []
    wins = []
    time_fast = 5 if isTraining else 30
    time_slow = 400

    for i in range(EPISODES):
        obs = env.reset()
        done = False
        episode_reward = 0

        if i % SHOW_EVERY == 0:
            print(f"on # {i}, epsilon: {epsilon}")
            if i != 0:
                print(f"{SHOW_EVERY} ep mean {np.mean(scores[-SHOW_EVERY:])}")
            show = True
        else:
            show = False

        while done == False:
          
            action = q.epsilon_greedy_policy(env, epsilon, obs)
            new_obs, reward, done, _ = env.step(action)
            episode_reward += reward

            if show:
                env.render(time_fast=time_fast, time_slow=time_slow)

            # if not done:
            q.update(env, obs, action, new_obs, reward) 
            
            if done:
                scores.append(episode_reward)
                wins.append(env.won)

            obs = new_obs       
          
        epsilon *= EPS_DECAY

    if SAVE_IMAGES:
        print("SAVING IMAGES")

    reward_title = 'q_learning_rewards_10x10' + NAME_COMPLEMENT
    if not isTraining: 
        reward_title += '_after_training'
    moving_avg = np.convolve(scores, np.ones((SHOW_EVERY,)) / SHOW_EVERY, mode='valid')
    plot_rewards(moving_avg, title=reward_title, save=SAVE_IMAGES)

    victories_title = 'q_learning_victories_10x10' + NAME_COMPLEMENT
    if not isTraining: 
        victories_title += '_after_training'
    victories_avg = np.convolve(wins, np.ones((SHOW_EVERY,)) / SHOW_EVERY, mode='valid')
    plot_victories(victories_avg, title=victories_title, save=SAVE_IMAGES)

    if (SAVE_TABLE):
        path = 'q_learning/q_tables/' + NAME_TABLE + '.pickle'
        print(f"SAVING {NAME_TABLE}")
        with open(path, "wb") as f:
          pickle.dump(q.q_table, f)

    env.close()

env = GameEnv(stochastic=isStochastic)
q_learning(env)