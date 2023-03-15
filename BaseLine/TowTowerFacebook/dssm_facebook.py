
"""
Date: create on 24/05/2022
References: 
    paper: (KDD'2020) Embedding-based Retrieval in Facebook Search
    url: https://arxiv.org/abs/2006.11632
Authors: Mincai Lai, laimincai@shanghaitech.edu.cn
"""

import torch
import torch.nn.functional as F
from layers import MLP

class FaceBookDSSM(torch.nn.Module):
    """Embedding-based Retrieval in Facebook Search
    It's a DSSM match model trained by hinge loss on pair-wise samples.
    Args:
        item_dict (dict[one-hot]): one-hot embedding for item_id.
        user_dict (dict[one-hot]): one-hot embedding for user_id.
        temperature (float): temperature factor for similarity score, default to 1.0.
        user_params (dict): the params of the User Tower module, keys include:`{"dims":list, "activation":str, "dropout":float, "output_layer":bool`}.
        item_params (dict): the params of the Item Tower module, keys include:`{"dims":list, "activation":str, "dropout":float, "output_layer":bool`}.
    """
    def __init__(self, args, user_params, item_params, item_dict, user_dict, temperature):
        super().__init__()

        self.args = args
        self.emb_output_size = args.emb_output_size
        self.item_dict = item_dict
        self.user_dict = user_dict
        self.temperature = temperature

        self.user_dims = args.m_user
        self.item_dims = args.n_item
        self.item_embeddings = torch.nn.Embedding(self.item_dims, embedding_dim=self.emb_output_size)
        self.user_embeddings = torch.nn.Embedding(self.user_dims, embedding_dim=self.emb_output_size)
        '''初始化embedding层的参数，mlp模型一般使用xavier_uniform_来初始化'''
        # torch.nn.init.normal_(self.user_embeddings.weight, std=0.1)
        # torch.nn.init.normal_(self.item_embeddings.weight, std=0.1)
        torch.nn.init.xavier_uniform_(self.item_embeddings.weight, gain=1)
        torch.nn.init.xavier_uniform_(self.user_embeddings.weight, gain=1)
        print('---use unnormal distribution initilizer---')

        self.all_items_emb = self.item_embeddings.weight.to(self.args.mdevice)
        self.all_users_emb = self.user_embeddings.weight.to(self.args.mdevice)

        self.item_mlp = MLP(self.item_dims, output_layer=False, **item_params)
        self.user_mlp = MLP(self.user_dims, output_layer=False, **user_params)

        self.mode = None

    '''获取item_id，user_id的one-hot向量表示'''
    def get_one_hot_emb(self, item_id, pos_user, neg_user):
        item_emb = self.item_dict[item_id[0]]
        pos_emb = self.user_dict[pos_user[0]]
        neg_emb = self.user_dict[neg_user[0]]

        for i in range(1, len(item_id)):
            item_emb = torch.cat([item_emb, self.item_dict[item_id[i]]])
            pos_emb = torch.cat([pos_emb, self.user_dict[pos_user[i]]])
            neg_emb = torch.cat([neg_emb, self.user_dict[neg_user[i]]])

        return item_emb.float(), pos_emb.float(), neg_emb.float()

    '''以item_tower为Query塔（查询塔），user_tower为Document塔（内容塔）'''
    def item_tower(self, item_id):
        if self.mode == "user":
            return None
        input_item = item_id    # 这里应该是one-hot向量表示item_id，大小为3846
        item_embedding = self.item_mlp(input_item)
        item_embedding = F.normalize(item_embedding, p=2, dim=1)
        return item_embedding


    def user_tower(self, pos_user, neg_user):
        if self.mode == "item":
            return None, None
        input_user_pos = pos_user   # 这里应该是one-hot向量表示pos_user_id，大小为1872
        if self.mode == "user":
            return self.user_mlp(input_user_pos), None

        input_user_neg = neg_user   # 这里应该是one-hot向量表示neg_user_id，大小为1872
        pos_embedding, neg_embedding = self.user_mlp(input_user_pos), self.user_mlp(input_user_neg)
        pos_embedding = F.normalize(pos_embedding, p=2, dim=1)
        neg_embedding = F.normalize(neg_embedding, p=2, dim=1)
        return pos_embedding, neg_embedding


    def forward(self, item_id, pos_user, neg_user):
        """得到item，pos_user，neg_user的embedding表示，将高维的one-hot向量变成低维的embedding向量"""
        item_embedding = self.item_tower(item_id)
        pos_user_embedding, neg_user_embedding = self.user_tower(pos_user, neg_user)

        if self.mode == "item":
            return item_embedding
        if self.mode == "user":
            return pos_user_embedding
        # calculate cosine score
        pos_score = torch.mul(item_embedding, pos_user_embedding).sum(dim=1)
        neg_score = torch.mul(item_embedding, neg_user_embedding).sum(dim=1)

        return pos_score, neg_score
