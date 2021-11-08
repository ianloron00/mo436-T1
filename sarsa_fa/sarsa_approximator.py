from sarsa_fa.basis.dependencies import *
from sarsa_fa.sarsa_extractor_features import *
import random

class Sarsa_Function_Approximator():
    def __init__(self, alpha=0.1, gamma=0.95, LAMBDA=0.2):
        self.ALPHA = alpha
        self.GAMMA = gamma
        self.LAMBDA = LAMBDA
        self.weights = {}
        self.features = {}
        self.extractor = Sarsa_Simple_Extractor()
        self.E = []
        
    def get_weights(self):
        return self.weights

    # overwrite
    def update (self, env, state, action, new_state, new_action, reward):
        # stochastic gradient descent update
        gradient = self.extractor.get_features(state, action)
        self.features = gradient

        new_q_val = self.get_q_value(new_state, new_action)
        q_val = self.get_q_value(state, action)

        target = reward + self.GAMMA * new_q_val
        
        delta_w = {}
        i = 0 
        for f in gradient:
            delta_w[f]  = (target - q_val)
            self.E[i] = (self.GAMMA*self.LAMBDA*self.E[i])+ gradient[f]
            self.weights[f] +=self.ALPHA *delta_w[f] * self.E[i]
            i+=1
        return 
    
 
    def get_q_value(self, state, action):
        
        q_val = 0
        features = self.extractor.get_features(state, action)

        for f in features:
            q_val += self.weights[f]*features[f]
        return q_val
    
    def initialize (self, env, name_weights='sarsa_weights.pickle'):
        if name_weights == None:
            
            print("INITIALIZING WEIGHTS")
            features = self.extractor.ini()

            #print(features)
            for f in features:
                self.weights[f] = np.random.random()
                self.E.append(0)
            
            '''
            print("INITIALIZING WEIGHTS")
            features = self.extractor.get_features(env.observation_space, env.action_space.sample())

            for f in features:
                self.weights[f] = np.random.random()  
                self.E.append(0)
            '''

    def epsilon_greedy_policy(self, env, epsilon, state):
    
        if random.uniform(0,1) < epsilon:
            return np.random.randint(0, env.action_space.n)
        else:
            return np.argmax([self.get_q_value(state, _action) for _action in range(env.action_space.n)])
    
    def get_max_q_val(self, env, state, action):
        actions = [a for a in range(env.action_space.n) if env.player.can_move(self, *env.player.get_direction(action))]
        np.argmax([self.get_q_value(state, _action) for _action in actions]) # range(env.action_space.n)])