from q_learning.basis.dependencies import *
from q_learning.q_learning import *
from q_learning.q_extractor_features import *

class Q_Function_Approximator(QLearning):
    def __init__(self, n_episodes=100000, alpha=0.1, gamma=0.95, epsilon=0.99, 
                 isTraining=False, name_table='q_learning_table'):
        super().__init__(n_episodes=n_episodes, alpha=alpha, gamma=gamma, epsilon=epsilon, 
                         isTraining=isTraining, name_table=name_table)
    
        self.weights = {}
        # vector features
        self.features = {}
        self.extractor = Q_Simple_Extractor()

    def get_weights(self):
        return self.weights

    # overwrite
    def update (self, env, state, action, new_state, reward):
        # stochastic gradient descent update
        gradient = self.extractor.get_features(state, action)
        self.features = gradient

        new_q_val = np.max([self.get_q_value(new_state, _action) for _action in range(env.action_space.n)])
        q_val = self.get_q_value(state, action)

        target = reward + self.GAMMA * new_q_val
        
        delta_w = {}
        for f in gradient:
            delta_w[f]  = self.ALPHA * (target - q_val) * gradient[f]
            self.weights[f] += delta_w[f]
    
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
        # should check if it is a possible action first?.
        features = self.extractor.get_features(state, action)

        for f in features:
            q_val += self.weights[f]*features[f]
        return q_val
    
    def initialize (self, env, name_weights='q_weights.pickle'):
        if name_weights == None:
            print("INITIALIZING WEIGHTS")
            features = self.extractor.get_features(env.observation_space, env.action_space.sample())

            for f in features:
                self.weights[f] = np.random.random()
        else:
            print("LOADING WEIGHTS")
            with open("q_learning/q_weights/" + name_weights, "rb") as f:
                self.weights = pickle.load(f)
            print(self.weights)

    def epsilon_greedy_policy(self, env, epsilon, state):
    
        if random.uniform(0,1) < epsilon:
            return np.random.randint(0, env.action_space.n)
        else:
            return np.argmax([self.get_q_value(state, _action) for _action in range(env.action_space.n)])
    
    def get_max_q_val(self, env, state, action):
        actions = [a for a in range(env.action_space.n) if env.player.can_move(self, *env.player.get_direction(action))]
        np.argmax([self.get_q_value(state, _action) for _action in actions]) # range(env.action_space.n)])