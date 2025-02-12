import torch.nn as nn
import torch
import torch.nn.functional as F


class MLPPolicy(nn.Module):
    def __init__(self, states: int, actions: int, net_arch: list[int]):
        super(MLPPolicy, self).__init__()

        layers = []
        layers.append(nn.Linear(states, net_arch[0]))
        layers.append(nn.ReLU())
        for i in range(1, len(net_arch)):
            layers.append(nn.Linear(net_arch[i - 1], net_arch[i]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(net_arch[-1], actions))
        self.model = nn.Sequential(*layers)

    def forward(self, x, prev_hidden=None):
        action_prob = self.model(x)
        return action_prob, None


class RNNPolicy(nn.Module):
    def __init__(
        self,
        states: int,
        actions: int,
        net_arch: list[int],
        hidden_size: int = 16,
    ):
        super(RNNPolicy, self).__init__()
        layers = []
        layers.append(nn.Linear(states, net_arch[0]))
        layers.append(nn.ReLU())
        for i in range(1, len(net_arch)):
            layers.append(nn.Linear(net_arch[i - 1], net_arch[i]))
            layers.append(nn.ReLU())
        self.seq = nn.Sequential(*layers)
        self.rnn = nn.GRU(net_arch[-1], hidden_size, batch_first=True)
        self.out_layer = nn.Linear(net_arch[-1], actions)
        self.hidden_size = hidden_size

    def forward(self, x, prev_hidden=None):
        if prev_hidden is None:
            prev_hidden = torch.zeros([1, self.hidden_size]).to(x.device)

        x = self.seq(x)
        x, hidden = self.rnn(x, prev_hidden)
        x = self.out_layer(x)
        return x, hidden
