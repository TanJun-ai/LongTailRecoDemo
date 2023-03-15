
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

class Item(torch.nn.Module):
    def __init__(self, config):
        super(Item, self).__init__()
        self.feature_dim = config['if_dim']
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

    def forward(self, x, vars=None):
        first_hidden = self.first_embedding_layer(x)
        first_hidden = F.relu(first_hidden)
        sec_hidden = self.second_embedding_layer(first_hidden)
        return F.relu(sec_hidden)

class User(torch.nn.Module):
    def __init__(self, config):
        super(User, self).__init__()
        self.feature_dim = config['uf_dim']
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

    def forward(self, x, vars=None):
        first_hidden = self.first_embedding_layer(x)
        first_hidden = F.relu(first_hidden)
        sec_hidden = self.second_embedding_layer(first_hidden)
        return F.relu(sec_hidden)
    
class Encoder(nn.Module):
    #Maps an (x_i, y_i) pair to a representation r_i.
    # Add the dropout into encoder ---03.31
    def __init__(self, x_dim, y_dim, h1_dim, h2_dim, z1_dim, dropout_rate):
        super(Encoder, self).__init__()

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.h1_dim = h1_dim
        self.h2_dim = h2_dim
        self.z1_dim = z1_dim
        self.dropout_rate = dropout_rate

        layers = [nn.Linear(self.x_dim + self.y_dim, self.h1_dim),
                  torch.nn.Dropout(self.dropout_rate),
                  nn.ReLU(inplace=True),
                  nn.Linear(self.h1_dim, self.h2_dim),
                  torch.nn.Dropout(self.dropout_rate),
                  nn.ReLU(inplace=True),
                  nn.Linear(self.h2_dim, self.z1_dim)]

        self.input_to_hidden = nn.Sequential(*layers)

    def forward(self, x, y):
        y = y.view(-1, 1)
        input_pairs = torch.cat((x, y), dim=1)
        return self.input_to_hidden(input_pairs)

class MuSigmaEncoder(nn.Module):
    def __init__(self, z1_dim, z2_dim, z_dim):
        super(MuSigmaEncoder, self).__init__()

        self.z1_dim = z1_dim
        self.z2_dim = z2_dim
        self.z_dim = z_dim
        self.z_to_hidden = nn.Linear(self.z1_dim, self.z2_dim)
        self.hidden_to_mu = nn.Linear(self.z2_dim, z_dim)
        self.hidden_to_logsigma = nn.Linear(self.z2_dim, z_dim)

    def forward(self, z_input):
        hidden = torch.relu(self.z_to_hidden(z_input))
        mu = self.hidden_to_mu(hidden)
        log_sigma = self.hidden_to_logsigma(hidden)
        std = torch.exp(0.5 * log_sigma)
        eps = torch.randn_like(std)
        z = eps.mul(std).add_(mu)
        return mu, log_sigma, z

class TaskEncoder(nn.Module):
    def __init__(self, x_dim, y_dim, h1_dim, h2_dim, final_dim, dropout_rate):
        super(TaskEncoder, self).__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.h1_dim = h1_dim
        self.h2_dim = h2_dim
        self.final_dim = final_dim
        self.dropout_rate = dropout_rate
        layers = [nn.Linear(self.x_dim + self.y_dim, self.h1_dim),
                  torch.nn.Dropout(self.dropout_rate),
                  nn.ReLU(inplace=True),
                  nn.Linear(self.h1_dim, self.h2_dim),
                  torch.nn.Dropout(self.dropout_rate),
                  nn.ReLU(inplace=True),
                  nn.Linear(self.h2_dim, self.final_dim)]

        self.input_to_hidden = nn.Sequential(*layers)

    def forward(self, x, y):
        y = y.view(-1, 1)
        input_pairs = torch.cat((x, y), dim=1)
        return self.input_to_hidden(input_pairs)

class MemoryUnit(nn.Module):
    # clusters_k is k keys
    def __init__(self, clusters_k, emb_size, temperature):
        super(MemoryUnit, self).__init__()
        self.clusters_k = clusters_k
        self.embed_size = emb_size
        self.temperature = temperature
        self.array = nn.Parameter(init.xavier_uniform_(torch.FloatTensor(self.clusters_k, self.embed_size)))

    def forward(self, task_embed):
        res = torch.norm(task_embed-self.array, p=2, dim=1, keepdim=True)
        res = torch.pow((res / self.temperature) + 1, (self.temperature + 1) / -2)
        # 1*k
        C = torch.transpose(res / res.sum(), 0, 1)
        # 1*k, k*d, 1*d
        value = torch.mm(C, self.array)
        # simple add operation
        new_task_embed = value + task_embed
        # calculate target distribution
        return C, new_task_embed

class Decoder(nn.Module):
    """
    Maps target input x_target and z, r to predictions y_target.
    """
    def __init__(self, x_dim, z_dim, task_dim, h1_dim, h2_dim, h3_dim, y_dim, dropout_rate):
        super(Decoder, self).__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.task_dim = task_dim
        self.h1_dim = h1_dim
        self.h2_dim = h2_dim
        self.h3_dim = h3_dim
        self.y_dim = y_dim
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(self.dropout_rate)

        self.hidden_layer_1 = nn.Linear(self.x_dim + self.z_dim, self.h1_dim)
        self.hidden_layer_2 = nn.Linear(self.h1_dim, self.h2_dim)
        self.hidden_layer_3 = nn.Linear(self.h2_dim, self.h3_dim)

        self.film_layer_1_beta = nn.Linear(self.task_dim, self.h1_dim, bias=False)
        self.film_layer_1_gamma = nn.Linear(self.task_dim, self.h1_dim, bias=False)
        self.film_layer_2_beta = nn.Linear(self.task_dim, self.h2_dim, bias=False)
        self.film_layer_2_gamma = nn.Linear(self.task_dim, self.h2_dim, bias=False)
        self.film_layer_3_beta = nn.Linear(self.task_dim, self.h3_dim, bias=False)
        self.film_layer_3_gamma = nn.Linear(self.task_dim, self.h3_dim, bias=False)

        self.final_projection = nn.Linear(self.h3_dim, self.y_dim)

    def forward(self, x, z, task):
        interaction_size, _ = x.size()
        z = z.unsqueeze(0).repeat(interaction_size, 1)
        # Input is concatenation of z with every row of x
        inputs = torch.cat((x, z), dim=1)
        hidden_1 = self.hidden_layer_1(inputs)
        beta_1 = torch.tanh(self.film_layer_1_beta(task))
        gamma_1 = torch.tanh(self.film_layer_1_gamma(task))
        hidden_1 = torch.mul(hidden_1, gamma_1) + beta_1
        hidden_1 = self.dropout(hidden_1)
        hidden_2 = F.relu(hidden_1)

        hidden_2 = self.hidden_layer_2(hidden_2)
        beta_2 = torch.tanh(self.film_layer_2_beta(task))
        gamma_2 = torch.tanh(self.film_layer_2_gamma(task))
        hidden_2 = torch.mul(hidden_2, gamma_2) + beta_2
        hidden_2 = self.dropout(hidden_2)
        hidden_3 = F.relu(hidden_2)

        hidden_3 = self.hidden_layer_3(hidden_3)
        beta_3 = torch.tanh(self.film_layer_3_beta(task))
        gamma_3 = torch.tanh(self.film_layer_3_gamma(task))
        hidden_final = torch.mul(hidden_3, gamma_3) + beta_3
        hidden_final = self.dropout(hidden_final)
        hidden_final = F.relu(hidden_final)

        y_pred = self.final_projection(hidden_final)
        return y_pred

