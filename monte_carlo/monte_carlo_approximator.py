
from numpy import float16
from dependencies import *
#from monte_carlo.monte_carlo import *
from monte_carlo.monte_carlo_extractor_features import MC_Simple_Extractor
from collections import defaultdict

class MC_Function_Approximator():
    def __init__(self , const_epsilon):
        self.weights = {}
        self.extractor = MC_Simple_Extractor()
        self.const_epsilon = const_epsilon
        self.rng = np.random.default_rng()
        

   
    def get_weights(self):
        return self.weights

    def update(self,episode,returns_sum,returns_count,states):
        sa_in_episode = set([(x[0], x[1]) for x in episode])
        s = 0
        for state, action in sa_in_episode:
            
            try:
                self.ns[state] += 1
            except:
                self.ns[state] = 1
            sa_pair = (state, action)
            
            first_occurence_idx = next(i for i,x in enumerate(episode)
                                       if x[0] == state and x[1] == action)
            G = sum([x[2]*(1) for i,x in enumerate(episode[first_occurence_idx:])])
         
            returns_sum[sa_pair] += G
            returns_count[sa_pair] += 1.0

            target = returns_sum[sa_pair] 
            #print(target)
            gradient = self.extractor.get_features(states[s], action)
            self.features = gradient
            
            q_val = self.get_q_value(states[s], action)
            delta_w = {}
            alpha = 1.0/returns_count[sa_pair]
            #print(len(gradient), len(self.weights))
            for f in gradient:
                delta_w[f] = alpha *(target - q_val) * gradient[f]

                self.weights[f] += delta_w[f]
            s+1
            returns_sum = defaultdict(float)
            returns_count = defaultdict(float)
        return returns_sum, returns_count

    """
    probably where the problem is!
    Look at how it calls the features extractor.
    """
    def get_q_value(self, state, action):
        """
        Q(s,a) = W^T * F (pg. 19 slide) 
        where W is the array of weights, and F is the vector features.
        """
        q_val = 0
        features = self.extractor.get_features(state, action)
      
        for f in features:
            q_val += self.weights[f]*features[f]
     
        return q_val

    def count_states(self):
        return len(self.ns.keys())

    def initialize(self, env, name_weights='mc_weights.pickle'):
        self.ns = {}
        self.nsa = {}
        if name_weights == None:
            print("INITIALIZING WEIGHTS")
            features = self.extractor.get_features(env.observation_space, env.action_space.sample())
            for f in features:
                self.weights[f] = np.random.random()

    
    def epsilon_greedy_policy(self, env, state):
     
        state_tuple = tuple([tuple(e) for e in state])
        try:
            epsilon = self.const_epsilon/(self.const_epsilon + self.ns[state_tuple])
            opt_action = np.argmax([self.get_q_value(state, _action) for _action in range(env.action_space.n)])   
            prob_actions = np.zeros(4) + epsilon/4
            prob_actions[opt_action] += 1 - epsilon
            action = self.rng.choice(4, p = prob_actions)
        except:
            return np.random.randint(0, env.action_space.n)
        return action
    '''
    def epsilon_greedy_policy(self, env, epsilon, state):
        
        if random.uniform(0,1) < epsilon:
            return np.random.randint(0, env.action_space.n)
        else:
            return np.argmax([self.get_q_value(state, _action) for _action in range(env.action_space.n)])
    '''
     
    def get_max_q_val(self, env, state, action):
        actions = [a for a in range(env.action_space.n) if env.player.can_move(self, *env.player.get_direction(action))]
        np.argmax([self.get_q_value(state, _action) for _action in actions]) # range(env.action_space.n)])
 