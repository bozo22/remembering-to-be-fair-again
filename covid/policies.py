import torch.nn as nn


class MLPPolicy(nn.Module):
    def __init__(self, states, actions):
        super(MLPPolicy, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(states, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, actions),
        )

    def forward(self, x):
        action_prob = self.model(x)
        return action_prob
