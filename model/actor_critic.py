import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal

class Actor(nn.Module): 
    # outputs the probability distribution of the actions given an observation

    def __init__(self, state_dim, action_dim, hidden_dim, cov_mat):
        super(Actor, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.cov_mat = cov_mat

    def forward(self, x):
        return MultivariateNormal(self.model(x), self.cov_mat)


class Critic(nn.Module):
    # outputs the value of the state given an observation

    def __init__(self, state_dim, hidden_dim):
        super(Critic, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.model(x)

