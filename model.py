import torch as T
import torch.nn as nn
from torch.distributions import Normal, Categorical


class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size, std=0.0, categoric=False):
        super(ActorCritic, self).__init__()

        self.categoric = categoric

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(num_inputs[2], 16, kernel_size=5, stride=2),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, kernel_size=5, stride=2),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=5, stride=2),
            nn.BatchNorm2d(32),
        )

        feature_out = self.feature_extractor(
                T.rand((1, num_inputs[2], num_inputs[0], num_inputs[1]))
        ).view(1,-1)

        self.critic = nn.Sequential(
            nn.Linear(feature_out.shape[1], hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        self.actor = nn.Sequential(
            nn.Linear(feature_out.shape[1], hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_outputs)
        )
        
        #random factor
        if not categoric:
            self.log_std = nn.Parameter(T.ones(1, num_outputs) * std)

    def forward(self, input):
        x = self.feature_extractor(input)
        x = x.view(input.shape[0],-1)
        value = self.critic(x)
        mu = self.actor(x)
        if not self.categoric:
            std = self.log_std.exp().expand_as(mu)
            dist = Normal(mu, std)
        else:
            dist = Categorical(mu)
        
        return dist, value