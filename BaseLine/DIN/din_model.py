
import torch
import torch.nn as nn
from torch.autograd import Variable


class Dice(nn.Module):
    """The Dice activation function mentioned in the `DIN paper
    https://arxiv.org/abs/1706.06978`
    """
    def __init__(self, epsilon=1e-3):
        super(Dice, self).__init__()
        self.epsilon = epsilon
        self.alpha = nn.Parameter(torch.randn(1))

    def forward(self, x: torch.Tensor):
        # x: N * num_neurons
        avg = x.mean(dim=1)  # N
        avg = avg.unsqueeze(dim=1)  # N * 1
        var = torch.pow(x - avg, 2) + self.epsilon  # N * num_neurons
        var = var.sum(dim=1).unsqueeze(dim=1)  # N * 1

        ps = (x - avg) / torch.sqrt(var)  # N * 1

        ps = nn.Sigmoid()(ps)  # N * 1
        return ps * x + (1 - ps) * self.alpha * x

'''base-model选择简单的MLP'''
class MLP(nn.Module):
    """Base-Model"""
    def __init__(self, args):
        super().__init__()

        self.args = args
        input_dim = args.input_dim
        first_layer_dim = args.first_embedding_dim
        second_layer_dim = args.second_embedding_dim

        '''定义模型的结构，3层MLP'''
        layers = list()

        layers.append(nn.Linear(input_dim, first_layer_dim))
        layers.append(nn.BatchNorm1d(first_layer_dim))
        layers.append(Dice())
        layers.append(nn.Dropout(p=self.args.dropout))

        layers.append(nn.Linear(first_layer_dim, second_layer_dim))
        layers.append(nn.BatchNorm1d(second_layer_dim))
        layers.append(Dice())
        layers.append(nn.Dropout(p=self.args.dropout))

        layers.append(nn.Linear(second_layer_dim, 1))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class DIN(nn.Module):
    """Deep Interest Network"""

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.n_item = args.n_item  # 3846
        self.m_user = args.m_user  # 1872
        self.emb_out_dim = self.args.emb_out_dim  # 64

        self._init_weight()
        self.mlp = MLP(self.args)

    '''初始化模型结构和参数'''
    def _init_weight(self):
        '''对items和users的one-hot编码进行embedding，统一输出self.latent_dim=64维'''
        self.embedding_item = nn.Embedding(
            num_embeddings=self.n_item, embedding_dim=self.emb_out_dim)
        self.embedding_user = nn.Embedding(
            num_embeddings=self.m_user, embedding_dim=self.emb_out_dim)
        # nn.init.normal_(self.embedding_user.weight, std=0.1)
        # nn.init.normal_(self.embedding_item.weight, std=0.1)
        nn.init.xavier_uniform_(self.embedding_user.weight, gain=1)
        nn.init.xavier_uniform_(self.embedding_item.weight, gain=1)
        print('use unnormal distribution initilizer')

    '''获取item，user的embedding'''
    def getEmbedding(self, items, users, history_users):
        all_items = self.embedding_item.weight.to(self.args.mdevice)
        all_users = self.embedding_user.weight.to(self.args.mdevice)

        items_emb = all_items[items].to(self.args.mdevice)
        user_emb = all_users[users].to(self.args.mdevice)
        hist_emb = all_users[history_users].to(self.args.mdevice)

        return items_emb, user_emb, hist_emb

    '''加入注意力机制'''
    def attention_layer(self, hist_emb, user_emb):

        total_hist_emb = []
        '''len(hist_emb)=2048（三维tensor），len(hist_emb[i])=19（二维tensor），
		len(hist_emb[i][j])=64，len(pos_emb[i])=64(一维tensor)，hist_score为19个分数，即每个user_id的权重
		hist_score=tensor([-0.0058, -0.0078,  0.0040, -0.0091, -0.0125, -0.0099,  0.0011,  0.0019,
		 0.0042, -0.0103, -0.0041,  0.0010, -0.0007, -0.0103, -0.0055, -0.0158, 0.0011,  0.0006,  0.0009]'''
        for i in range(len(hist_emb)):  # 对每一批进行运算，这里的每一批大小为2048
            # for j in range(len(hist_emb[i])):   # 对每一批（2048）中的每一条历史数据（包含19个user_id）进行运算
            # 计算每一条历史记录的分数，包含了里面19个user_id与候选的pos_user_id的相似度分数（使用内积的方法求得）
            hist_score = torch.mul(hist_emb[i], user_emb[i])
            hist_score = torch.sum(hist_score, dim=1)

            '''初始化一个embedding，用来计算19个user_id的embedding加权和，
			即获得19个user_id的embedding加权向量total_emb,这个向量的大小也是64维'''
            total_emb = torch.tensor(0).to(self.args.mdevice)
            for u in range(len(hist_emb[i])):
                emb = torch.mul(hist_emb[i][u], hist_score[u])
                total_emb = total_emb + emb

            total_hist_emb.append(total_emb.tolist())

        item_his_emb = torch.tensor(total_hist_emb).to(self.args.mdevice)

        return item_his_emb

    '''输入的数据包含一个最小批（这里设置为248）的字段为：items, pos, neg, history'''
    def forward(self, items, users, history_users):

        items_emb, user_emb, hist_emb = self.getEmbedding(items, users, history_users)
        user_his_emb = self.attention_layer(hist_emb, user_emb)

        x = torch.cat((items_emb, user_his_emb, user_emb), 1)

        return self.mlp(x)