import torch
from torch import nn


class TextGate(nn.Module):

    def __init__(
        self,
        ts_dim,
        text_dim,
        hidden_dim=128
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(ts_dim+text_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,1),
            nn.Sigmoid()
        )


    def forward(
        self,
        ts_feature,
        text_feature
    ):

        x=torch.cat(
            [
                ts_feature,
                text_feature
            ],
            dim=-1
        )

        gate=self.net(x)

        return gate