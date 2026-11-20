import torch
import torch.nn as nn


class ChannelEstimator(nn.Module):

    def __init__(self):
        super(ChannelEstimator, self).__init__()

        self.network = nn.Sequential(

            nn.Linear(32, 128),

            nn.ReLU(),

            nn.Linear(128, 64),

            nn.ReLU(),

            nn.Linear(64, 2)
        )

    def forward(self, x):

        return self.network(x)