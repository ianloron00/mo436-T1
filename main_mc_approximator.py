import numpy as np
from monte_carlo.basis.environment import * 
from monte_carlo.basis.dependencies import *
from monte_carlo.basis.graphs import *
# from graphs import plot_rewards, plot_victories
import pickle
from monte_carlo.monte_carlo_approximator import MC_Function_Approximator
from monte_carlo.monte_carlo_extractor_features import *
from collections import defaultdict
isTraining = True
isStochastic = True
SAVE_IMAGES = True
#EPSILON = 0.99 if isTraining else 0.005

EPISODES = 100 if isTraining else 10
CONST_EPSILON = 10 if isTraining else 0.005
EPS_DECAY = 2**(-4 / EPISODES) if isTraining else 1

SHOW_EVERY = int(EPISODES/10) if isTraining else 1

NAME_COMPLEMENT = '_stochastic' if isStochastic else '_deterministic'
NAME_WEIGHTS = 'mc_weights_10x10' + NAME_COMPLEMENT

SAVE_WEIGHTS = True if isTraining else False

N_MAX_STEPS = 200

start_weights = None if isTraining else NAME_WEIGHTS + '.pickle'

def mc_function_approximator(env):
    mc = MC_Function_Approximator(const_epsilon=CONST_EPSILON)
    mc.initialize(env, name_weights=start_weights)
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    states = []
    scores = []
    wins = []
    time_fast = 5 if isTraining else 30
    time_slow = 600
    #epsilon = EPSILON
    for i in range(EPISODES):
        obs = env.reset()
        episode = []
      
        done = False
        episode_reward = 0
        n_steps = 0
 
        if i % SHOW_EVERY == 0:
            print(f"on # {i}, the known states are {mc.count_states()}")
            if i != 0:
                print(f"{SHOW_EVERY} ep mean {np.mean(scores[-SHOW_EVERY:])}")
            show = False
        else:
            show = False
        
        while done == False and n_steps< N_MAX_STEPS:
            n_steps+=1

            if isTraining:
                action = mc.epsilon_greedy_policy(env, obs)
            else:
                action = mc.greedy_policy(env,obs)

            new_obs, reward, done, _ = env.step(action)
            episode_reward += reward

            obs_tuple = tuple([tuple(e) for e in obs])
            episode.append((obs_tuple,action,reward))
            states.append(obs)
            if show:
                env.render(time_fast=time_fast, time_slow=time_slow)
            
            if done:
                scores.append(episode_reward)
                wins.append(env.won)

            obs = new_obs
        returns_sum, returns_count =  mc.update(episode, returns_sum,returns_count, states)
        #epsilon *= EPS_DECAY    
    reward_title = 'mc_approximator_rewards_10x10' + NAME_COMPLEMENT
    if not isTraining: 
        reward_title += '_after_training'
    moving_avg = np.convolve(scores, np.ones((SHOW_EVERY,)) / SHOW_EVERY, mode='valid')
    plot_rewards(moving_avg, reward_title, save=SAVE_IMAGES)

    victories_title = 'mc_approximator_victories_10x10' + NAME_COMPLEMENT
    if not isTraining: 
        victories_title += '_after_training'
    victories_avg = np.convolve(wins, np.ones((SHOW_EVERY,)) / SHOW_EVERY, mode='valid')
    plot_victories(victories_avg, victories_title, save=SAVE_IMAGES)

    if (SAVE_WEIGHTS):
        path = 'monte_carlo/monte_carlo_weights/' + NAME_WEIGHTS + '.pickle'
        with open(path, "wb") as f:
          pickle.dump(mc.get_weights, f)

    env.close()

env = GameEnv(tabular=False, stochastic=isStochastic)
mc_function_approximator(env)