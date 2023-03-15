
# coding: utf-8
# author: lu yf
# create date: 2019-12-10 14:22

import torch
from torch.autograd import Variable
import torch.nn.functional as F

# new movielens-1m dataset embedding
'''one-hot向量转变为embedding向量，即在lastfm-20中，
item的one-hot有3846维，user的one-hot有1872维，变成embedding后，统一变成16维'''
class Item(torch.nn.Module):
    def __init__(self, config):
        super(Item, self).__init__()
        self.feature_dim = config['item_dim']
        self.first_embedding_dim = config['first_embedding_dim']    # 32
        self.second_embedding_dim = config['second_embedding_dim']  # 16

        self.first_embedding_layer = torch.nn.Linear(
            in_features=self.feature_dim,
            out_features=self.first_embedding_dim,
            bias=True
        )

        self.second_embedding_layer = torch.nn.Linear(
            in_features=self.first_embedding_dim,
            out_features=self.second_embedding_dim,
            bias=True
        )

    def forward(self, x):
        first_hidden = self.first_embedding_layer(x)
        first_hidden = F.relu(first_hidden)
        sec_hidden = self.second_embedding_layer(first_hidden)
        return F.relu(sec_hidden)

class User(torch.nn.Module):
    def __init__(self, config):
        super(User, self).__init__()
        self.feature_dim = config['user_dim']
        self.first_embedding_dim = config['first_embedding_dim']
        self.second_embedding_dim = config['second_embedding_dim']

        self.first_embedding_layer = torch.nn.Linear(
            in_features=self.feature_dim,
            out_features=self.first_embedding_dim,
            bias=True
        )

        self.second_embedding_layer = torch.nn.Linear(
            in_features=self.first_embedding_dim,
            out_features=self.second_embedding_dim,
            bias=True
        )

    def forward(self, x):
        first_hidden = self.first_embedding_layer(x)
        first_hidden = F.relu(first_hidden)
        sec_hidden = self.second_embedding_layer(first_hidden)
        return F.relu(sec_hidden)
