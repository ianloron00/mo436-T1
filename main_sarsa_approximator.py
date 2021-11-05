
import numpy as np
from sarsa_fa.basis.environment import * 
from sarsa_fa.basis.dependencies import *
from sarsa_fa.basis.graphs import *
import pickle
from sarsa_fa.sarsa_approximator import Sarsa_Function_Approximator
from sarsa_fa.sarsa_extractor_features import *

isTraining = True
isStochastic = True
SAVE_IMAGES = True

GAMMA = 0.95 if isTraining else 0.0
EPISODES = 10000 if isTraining else 10
EPSILON = 0.99 if isTraining else 0.005
EPS_DECAY = 2**(-4 / EPISODES) if isTraining else 1

SHOW_EVERY = int(EPISODES/10) if isTraining else 1

NAME_COMPLEMENT = '_stochastic' if isStochastic else '_deterministic'
NAME_WEIGHTS = 'sarsa_weights_10x10' + NAME_COMPLEMENT

SAVE_WEIGHTS = True if isTraining else False

N_MAX_STEPS = 650

start_weights = None if isTraining else NAME_WEIGHTS + '.pickle'

def sarsa_function_approximator(env):
    sarsa_fa = Sarsa_Function_Approximator(gamma=GAMMA)
    sarsa_fa.initialize(env, name_weights=start_weights)

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
        
        action = sarsa_fa.epsilon_greedy_policy(env, epsilon, obs)
    
        while done == False and n_steps < N_MAX_STEPS:
            n_steps += 1

            new_obs, reward, done, _ = env.step(action)
            new_action = sarsa_fa.epsilon_greedy_policy(env, epsilon, new_obs)
            episode_reward += reward
            
            if show:
                env.render(time_fast=time_fast, time_slow=time_slow)

            sarsa_fa.update(env, obs, action, new_obs, new_action, reward) 
            
            if done:
                scores.append(episode_reward)
                wins.append(env.won)

            action = new_action
            obs = new_obs       
          
        epsilon *= EPS_DECAY

    reward_title = 'sarsa_approximator_rewards_10x10' + NAME_COMPLEMENT
    if not isTraining: 
        reward_title += '_after_training'
    moving_avg = np.convolve(scores, np.ones((SHOW_EVERY,)) / SHOW_EVERY, mode='valid')
    plot_rewards(moving_avg, reward_title, save=SAVE_IMAGES)

    victories_title = 'sarsa_approximator_victories_10x10' + NAME_COMPLEMENT
    if not isTraining: 
        victories_title += '_after_training'
    victories_avg = np.convolve(wins, np.ones((SHOW_EVERY,)) / SHOW_EVERY, mode='valid')
    plot_victories(victories_avg, victories_title, save=SAVE_IMAGES)

    if (SAVE_WEIGHTS):
        path = 'sarsa_fa/sarsa_weights/' + NAME_WEIGHTS + '.pickle'
        with open(path, "wb") as f:
          pickle.dump(sarsa_fa.get_weights, f)

    env.close()

env = GameEnv(tabular=False, stochastic=isStochastic)
sarsa_function_approximator(env)