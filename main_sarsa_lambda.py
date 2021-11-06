import os
from datetime import datetime
import numpy as np
from sarsa_lambda.basis.environment import *
from sarsa_lambda.basis.graphs import *
# from graphs import plot_rewards, plot_victories
import pickle
from sarsa_lambda.basis.dependencies import *
from sarsa_lambda.sarsa_lambda import SarsaLambda

isTraining = True
isStochastic = True
SAVE_IMAGES = True
PATH = '.data'

EPISODES = 150_000 if isTraining else 10
GAMMA = 0.10 if isTraining else 0.0
CONST_EPSILON = [100, 1000, 10000] if isTraining else 0
LAMBDA = [0, 0.2, 0.4, 0.6, 0.8, 1]

SHOW_EVERY = int(EPISODES/20) if isTraining else 1

NOW = datetime.now().strftime('%Y%m%d_%H%M%S')
NAME_COMPLEMENT = '_stochastic' if isStochastic else '_deterministic'
NAME_Q_TABLE = 'sarsa_lambda_Q_table' + NAME_COMPLEMENT + NOW
NAME_SCORES = 'sarsa_lambda_Scores' + NAME_COMPLEMENT + NOW
NAME_NSA_TABLE = 'sarsa_lambda_NSA_table' + NAME_COMPLEMENT + NOW
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

def sarsa_lambda(env):
    for lambd in LAMBDA:
        for const_eps in CONST_EPSILON:
            print(f"lambda  = {lambd} and N0 = {const_eps}")
            agent = SarsaLambda(gamma=GAMMA, const_epsilon=const_eps, lambd = lambd)
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

                if isTraining:
                    action = agent.epsilon_greedy_policy(env, obs)
                else:

                    action = agent.greedy_policy(env, obs)

                while done == False:

                    new_obs, reward, done, _ = env.step(action)
                    episode_reward += reward

                    if not done:
                        if isTraining:
                            new_action = agent.epsilon_greedy_policy(env, obs)
                        else:

                            new_action = agent.greedy_policy(env, obs)

                    agent.update(env, obs, action, new_obs, new_action, reward, done)

                    if show:
                        env.render(time_fast=time_fast, time_slow=time_slow)

                    if done:
                        scores.append(episode_reward)
                        wins.append(env.won)

                    obs = new_obs
                    action = new_action

            if SAVE_IMAGES:
                print("SAVING IMAGES")

            reward_title = 'sarsa_lambda_rewards_10x10' + NAME_COMPLEMENT
            if not isTraining:
                reward_title += '_after_training'
            moving_avg = np.convolve(scores, np.ones((SHOW_EVERY,)) / SHOW_EVERY, mode='valid')
            plot_rewards(moving_avg, title=reward_title, lambd = lambd, N0 = const_eps, save=SAVE_IMAGES)

            victories_title = 'sarsa_lambda_victories_10x10' + NAME_COMPLEMENT
            if not isTraining:
                victories_title += '_after_training'
            victories_avg = np.convolve(wins, np.ones((SHOW_EVERY,)) / SHOW_EVERY, mode='valid')
            plot_victories(victories_avg, title=victories_title, lambd = lambd, N0 = const_eps, save=SAVE_IMAGES)

            if (SAVE_TABLE):
                path = 'sarsa_lambda/q_tables/' + NAME_Q_TABLE + '_lambda_' + str(lambd) + '_NO_' + str(const_eps) + '.pickle'
                path2 = 'sarsa_lambda/scores/' + NAME_SCORES + '_lambda_' + str(lambd) + '_NO_' + str(const_eps) + '.pickle'
                print(f"SAVING {NAME_Q_TABLE}")
                with open(path, "wb") as f:
                  pickle.dump(agent.q_table, f)
                with open(path2, "wb") as f:
                  pickle.dump(scores, f)

    env.close()

def main():
    env = GameEnv(stochastic=isStochastic)
    sarsa_lambda(env)

if __name__ == '__main__':
    main()
