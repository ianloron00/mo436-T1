import numpy as np  
import torch  
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, learning_rate):
        super(ActorCritic, self).__init__()

        self.num_actions = num_actions
        self.critic_linear1 = nn.Linear(num_inputs, hidden_size)
        self.critic_linear2 = nn.Linear(hidden_size, hidden_size)
        self.critic_linear3 = nn.Linear(hidden_size, hidden_size)
        self.critic_linear4 = nn.Linear(hidden_size, 1)

        self.actor_linear1 = nn.Linear(num_inputs, hidden_size)
        self.actor_linear2 = nn.Linear(hidden_size, hidden_size)
        self.actor_linear3 = nn.Linear(hidden_size, hidden_size)
        self.actor_linear4 = nn.Linear(hidden_size, num_actions)
        
    def forward(self, state):
        state = Variable(torch.from_numpy(state).float().unsqueeze(0))
        value = F.relu(self.critic_linear1(state))
        value = self.critic_linear2(value)
        value = self.critic_linear3(value)
        value = self.critic_linear4(value)
        
        policy_dist = F.relu(self.actor_linear1(state))
        policy_dist = self.actor_linear2(policy_dist)
        policy_dist = self.actor_linear3(policy_dist)
        policy_dist = F.softmax(self.actor_linear4(policy_dist), dim=1)
    
        return value, policy_dist
