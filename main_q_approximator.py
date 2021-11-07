# /home/ianloron00/Desktop/Unicamp/1s_2021/ML/MachineLearning/P3/198933_215076/RL

import numpy as np
from q_learning.basis.environment import * 
from q_learning.basis.dependencies import *
from q_learning.basis.graphs import *
# from graphs import plot_rewards, plot_victories
import pickle
from q_learning.q_learning import QLearning
from q_learning.q_approximator import Q_Function_Approximator
from q_learning.q_extractor_features import *

isTraining = False
isStochastic = True
SAVE_IMAGES = False

GAMMA = 0.95 if isTraining else 0.0
EPISODES = 50 if isTraining else 10
EPSILON = 0.99 if isTraining else 0.005
# achieves half of epsilon at 1/K-th of the number of timesteps.
# decay = 2**(-log2(0.5/EPSILON) / (EPISODES / K)) ~ 2**(-1 / (EPISODES / K)) ~ 2**(-K / EPISODES)
EPS_DECAY = 2**(-4 / EPISODES) if isTraining else 1

SHOW_EVERY = int(EPISODES/10) if isTraining else 1

NAME_COMPLEMENT = '_stochastic' if isStochastic else '_deterministic'
NAME_WEIGHTS = 'q_weights_10x10' + NAME_COMPLEMENT

SAVE_WEIGHTS = True if isTraining else False

N_MAX_STEPS = 650

start_weights = None if isTraining else NAME_WEIGHTS + '.pickle'

def q_function_approximator(env):
    q = Q_Function_Approximator(gamma=GAMMA)
    q.initialize(env, name_weights=start_weights)

    epsilon = EPSILON
    scores = []
    wins = []
    time_fast = 5 if isTraining else 30
    time_slow = 600

    for i in range(EPISODES):
        obs = env.reset()
        done = False
        episode_reward = 0
        n_steps = 0

        if i % SHOW_EVERY == 0:
            print(f"on # {i}, epsilon: {epsilon}")
            if i != 0:
                print(f"{SHOW_EVERY} ep mean {np.mean(scores[-SHOW_EVERY:])}")
            show = True
        else:
            show = False
        #### comment it to show display ###
        show = False

        while done == False and n_steps < N_MAX_STEPS:
            n_steps += 1

            action = q.epsilon_greedy_policy(env, epsilon, obs)
            new_obs, reward, done, _ = env.step(action)
            episode_reward += reward

            if show:
                env.render(time_fast=time_fast, time_slow=time_slow)

            q.update(env, obs, action, new_obs, reward) 
            
            if done:
                scores.append(episode_reward)
                wins.append(env.won)

            obs = new_obs       
          
        epsilon *= EPS_DECAY

    reward_title = 'q_approximator_rewards_10x10' + NAME_COMPLEMENT
    if not isTraining: 
        reward_title += '_after_training'
    moving_avg = np.convolve(scores, np.ones((SHOW_EVERY,)) / SHOW_EVERY, mode='valid')
    plot_rewards(moving_avg, reward_title, save=SAVE_IMAGES)

    victories_title = 'q_approximator_victories_10x10' + NAME_COMPLEMENT
    if not isTraining: 
        victories_title += '_after_training'
    victories_avg = np.convolve(wins, np.ones((SHOW_EVERY,)) / SHOW_EVERY, mode='valid')
    plot_victories(victories_avg, victories_title, save=SAVE_IMAGES)

    if (SAVE_WEIGHTS):
        path = 'q_learning/q_weights/' + NAME_WEIGHTS + '.pickle'
        with open(path, "wb") as f:
          pickle.dump(q.weights, f, protocol=pickle.HIGHEST_PROTOCOL)

    env.close()

env = GameEnv(tabular=False, stochastic=isStochastic)
q_function_approximator(env)