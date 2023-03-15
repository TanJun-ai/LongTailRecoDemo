
from scipy.sparse import csr_matrix
import scipy.sparse as sp
import numpy as np
import torch
import torch
from torch import nn, optim
import numpy as np
from time import time
import argparse
from dataset_load import UniformSample
from utils import shuffle, minibatch

# '''稀疏矩阵转成稠密矩阵'''
# def _convert_sp_mat_to_sp_tensor(X):
#     coo = X.tocoo().astype(np.float32)
#     row = torch.Tensor(coo.row).long()
#     col = torch.Tensor(coo.col).long()
#     index = torch.stack([row, col])
#     data = torch.FloatTensor(coo.data)
#     return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

# a = torch.tensor(1)
# print(a)
# print(a.item())
# S = [[1, 2], [3, 4]]
# r = [5, 6]
# S.append(r)
# print(S)
# S.remove(S[0])
# print(S)
#
# R = sp.dok_matrix((3, 5), dtype=np.float32)
#
# R[0, 0] = 1
# R[0, 1] = 1
# R[0, 2] = 1
# R[1, 0] = 1
# R[1, 1] = 1
# R[1, 4] = 1
# R[2, 2] = 1
# R[2, 3] = 1
# R[2, 4] = 1
# R = R.tolil()
#
#
# adj_mat = sp.dok_matrix((8, 8), dtype=np.float32)
# adj_mat = adj_mat.tolil()
# adj_mat[:3, 3:] = R
# adj_mat[3:, :3] = R.T
# print("----------adj_mat-----------")
# print(adj_mat)
#
# # atten = R.T.dot(R)
# # atten = R.dot(R.T)
# atten = adj_mat.T.dot(adj_mat)
# print("==========atten=========")
# print(atten)
# atten_row_sum = np.array(atten.sum(axis=1))
# print("--------atten_row_sum-------")
# print(atten_row_sum)
# a_inv = np.power(atten_row_sum, -0.5).flatten()
# a_inv[np.isinf(a_inv)] = 0.
# a_mat = sp.diags(a_inv)
# print("--------a_mat-------")
# print(a_mat)
# s_adj = a_mat.dot(adj_mat)
# s_adj = s_adj.dot(a_mat)
# s_adj = s_adj.tocsr()
# print("===============s_adj===============")
# print(s_adj)
# # atten_mat = sp.dok_matrix((6, 6), dtype=np.float32)
# # atten_mat = atten_mat.tolil()
# # atten_mat[:3, 3:] = atten
# # atten_mat[3:, :3] = atten.T
#
# # a_inv = np.power(atten_row_sum, -0.5).flatten()
# # a_inv[np.isinf(a_inv)] = 0.
# # a_mat = sp.diags(a_inv)
# # print("===========a_mat==========")
# # print(a_mat)
# # atten_adj = a_mat.dot(atten_mat)
# # atten_adj = atten_adj.dot(a_mat)
# # atten_adj = atten_adj.tocsr()
# # print("-----------------atten_adj---------------------")
# # print(atten_adj)
# # for i in range(3):
# #     for j in range(3):
# #         adj_mat[i, j] = i + j    # Update element
# # print("---------adj_mat-----------")
# # print(adj_mat.shape)
# # print(adj_mat)
#
# row_sum = np.array(adj_mat.sum(axis=1))
# print("---------row_sum-----------")
# print(row_sum)
# d_inv = np.power(row_sum, -0.5).flatten()
# d_inv[np.isinf(d_inv)] = 0.
# d_mat = sp.diags(d_inv)
# print("---------------d_inv----------")
# print(d_inv)
# print("=========d_mat========")
# print(d_mat)
#
# norm_adj = d_mat.dot(adj_mat)
# norm_adj = norm_adj.dot(d_mat)
# norm_adj = norm_adj.tocsr()
#
# I_mat = sp.eye(adj_mat.shape[0])
# # print("---------adj_mat.shape[0]----------")
# # print(adj_mat.shape[0])
# # print("=======I_mat=======")
# # print(I_mat)
#
# # d_inv = [0.3, 0., 0.2]
# # d_mat = sp.diags(d_inv)
# # norm_adj = d_mat.dot(adj_mat)
# print("=========norm_adj========")
# print(norm_adj.shape)
# print(norm_adj)
# # print("=========norm_adj========")
# # print(norm_adj)
#
# Graph = _convert_sp_mat_to_sp_tensor(norm_adj)
# Graph = Graph.coalesce()
#
# all_emb = torch.tensor(
# 	   [[-0.01,  0.03,  0.01],
#         [0.03,  0.02,  0.01],
#         [-0.03, -0.02,  0.01],
#         [-0.03,  0.01,  0.05],
#         [-0.03, -0.05,  0.02],
#         [-0.04, -0.01,  0.02],
# 	    [-0.02, -0.01,  0.02],
# 	    [-0.03, -0.01,  0.02]])
# embeddings = [all_emb]
# print("---------==all_emb==------------")
# print(all_emb)
# print("=============Graph===============")
# print(Graph)
# for layer in range(3):  # 3层
# 	all_emb = torch.sparse.mm(Graph, all_emb)
# 	# print("---------Graph------------")
# 	# print(Graph)
# 	print("---------all_emb------------")
# 	print(all_emb)
#
# 	embeddings.append(all_emb)
#
# # print("------------Graph------------")
# # print(Graph)


# '''将int型转成one-hot向量'''
# def to_onehot_dict(_list):
#
#     _tensor = []
#     length = len(_list)
#     for index, element in enumerate(_list):
#         _vector = torch.zeros(1, length).long()
#         element = int(element)
#         _vector[:, element] = 1.0
#         _tensor.extend(_vector.tolist())
#
#     total_tensor = torch.tensor(_tensor)
#     return total_tensor
#
# tse_list = []
# for i in range(3846):
#     tse_list.append(i)
#
# one_hot_tensor = to_onehot_dict(tse_list)
# # print("------------one_hot_tensor-------------")
# # print(one_hot_tensor.shape)
# # print(one_hot_tensor)
#
# test_pos_user = 634
# train_list = [258, 287, 296, 327, 345, 349, 409, 501, 503, 508, 523, 529, 563, 564, 596, 605, 633, 637,
#               638, 664, 702, 718, 740, 746, 768, 831, 839, 842, 873, 902, 910, 935, 940, 966, 994]
#
# def attention_scores(hist_list, pos_user):
#     score_list = []
#     for i in range(len(hist_list)):
#         dist = torch.sqrt(torch.pow(torch.tensor(pos_user-hist_list[i]), 2))
#         sim = 1/(1 + dist)
#         score_list.append(sim.item())
#     sim_scores = torch.tensor(score_list)
#     _scores, recom_indexs = sim_scores.view(-1).sort(descending=True)
#
#     return _scores[0].item(), recom_indexs[0].item()
#
#
# sim_score, recom_index = attention_scores(train_list, test_pos_user)
# print("--------sim_score-------")
# print(sim_score)
# print("--------recom_index-------")
# print(recom_index)
# print("--------train_list-------")
# print(train_list[recom_index])
#
# print("---------test_pos_user-train_list[0]---------")
# print(test_pos_user-train_list[0])
# dist = torch.pow(torch.tensor(test_pos_user-train_list[0]), 2)
# truth_dist = torch.sqrt(dist)
# print("---------truth_dist---------")
# print(truth_dist)
# sim = 1/(1 + truth_dist)
# print(sim.item())
#
# embedding_item = nn.Embedding(num_embeddings=1872, embedding_dim=64)
# embedding_user = nn.Embedding(num_embeddings=3846, embedding_dim=64)
# nn.init.normal_(embedding_user.weight, std=0.1)
# nn.init.normal_(embedding_item.weight, std=0.1)
# item_emb = embedding_item.weight
# user_emb = embedding_user.weight
#
# # print("----------embedding_user------------")
# # print(user_emb.shape)
# # print(user_emb)
#
# hist_user_emb = user_emb[train_list]
# pos_user_emb = user_emb[test_pos_user]
#
# each_score = torch.mul(hist_user_emb, pos_user_emb)   # 计算每个hits_user与user的分数
# each_score = torch.sum(each_score, dim=1)
# _score, recom_index = each_score.view(-1).sort(descending=True)
#
# print("------_score--------")
# print(_score)
# print("------recom_index--------")
# print(recom_index)



class LoaderDataset:
    """
    Procceeding Dataset for lastfm20 ：这个数据集分成support-set 和 query-set，
    support-set 中的item最大值为3846，user最大值为1872，
    query-set 中的item最大值为3846，user最大值为1872
    """
    def __init__(self, parameter, train_path, dataset):
        self.n_item = 0
        self.m_user = 0
        self.trainDataSize = 0

        self.save_path = parameter.data_dir + dataset
        self.args = parameter
        trainUniqueItems, trainItem, trainUser = [], [], []

        with open(train_path) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip().strip('\n').split(' ')
                    users = [int(i) for i in l[1:]]
                    item_id = int(l[0])
                    trainUniqueItems.append(item_id)
                    trainItem.extend([item_id] * len(users))
                    trainUser.extend(users)
                    self.n_item = max(self.n_item, item_id)
                    self.m_user = max(self.m_user, max(users))
                    self.trainDataSize += len(users)

        self.trainUniqueItems = np.array(trainUniqueItems)
        self.trainItem = np.array(trainItem)
        self.trainUser = np.array(trainUser)

        print(f"{self.trainDataSize} interactions for training")

        '''手动设置item和user的最大容量，确保矩阵不会越界'''
        self.n_item = self.args.n_item
        self.m_user = self.args.m_user
        '''user 和正样本item 的交互矩阵，矩阵的元素值都为1.0，这里从0开始，即user_id为0的user其
        真实的user_id=1,(0, 1) 1.0。。。(6039, 3819)	1.0,数据类型为矩阵csr_matrix'''
        self.UserItemNet = csr_matrix((np.ones(len(self.trainItem)), (self.trainItem, self.trainUser)),
                                      shape=(self.n_item, self.m_user))

        self.users_D = np.array(self.UserItemNet.sum(axis=1)).squeeze()
        self.users_D[self.users_D == 0.] = 1.
        self.items_D = np.array(self.UserItemNet.sum(axis=0)).squeeze()
        self.items_D[self.items_D == 0.] = 1.

        '''self.n_user=6040，拿到每个item对应的正样本user_id的list，self.allPos=[...,[...3671 3683 3703 3735 3751 3819]]'''
        self.allPos = self.getUserPosItems(list(range(self.n_item)))

        print(f"{parameter.dataset_name} is ready to go")

        self.Graph = None

    def getUserPosItems(self, items):
        posUsers = {}
        for item in items:
            posUsers[item] = self.UserItemNet[item].nonzero()[1]
        return posUsers

    '''稀疏矩阵转成稠密矩阵'''
    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    '''如果已经有了生成好的图卷积层，直接加载，如果没有，需要重新生成
    生成原理需要对照论文中的公式，这里的创新点是在其基础上增加候选user和交互历史user的相似度权重'''
    def getSparseGraph(self):
        # print("loading exist adjacency matrix")
        # try:
        #     pre_adj_mat = sp.load_npz(self.save_path + '/s_pre_adj_mat.npz')
        #     print("successfully loaded......")
        #     norm_cos_adj = pre_adj_mat
        # except :
        print("generating new adjacency matrix......")
        start = time()

        adj_mat = sp.dok_matrix((self.n_item + self.m_user, self.n_item + self.m_user), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.UserItemNet.tolil()
        '''adj_mat对应公式（8）中的邻接矩阵A'''
        adj_mat[:self.n_item, self.n_item:] = R
        adj_mat[self.n_item:, :self.n_item] = R.T
        adj_mat = adj_mat.todok()
        np.seterr(divide='ignore', invalid='ignore')  # 消除被除数为0的警告
        row_sum = np.array(adj_mat.sum(axis=1))  # 将adj_mat中一行的元素相加

        '''这里的权重仅仅是距离的权重，距离越远，权重越低，创新点是在这个基础上加入相似度的权重
        norm_adj=1/√|Nu||Ni|，norm_adj=norm_adj+γ，γ是用attention原理计算得到,
        norm_adj是一个(5718, 5718)的矩阵，d_mat和adj_mat也是(5718, 5718)的矩阵，在lastfm数据集上，
        R的形状为(3846, 1872)，即5718=3846+1872，row_sum形状为（5718,1）'''
        d_inv = np.power(row_sum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        '''形成对角矩阵，比如d_inv=[0.2,0,0.1]，会形成一个3x3的对角矩阵，对角元素分别是0.2,0,0.1，
        这里因为d_inv为size=5718的list，因此对角矩阵d_mat的形状为(5718, 5718)'''
        d_mat = sp.diags(d_inv)

        '''d_mat对应公式（8）中的矩阵D^(-1/2)，下面的操作实现公式（8）：D^(-1/2)*A*D^(-1/2)'''
        norm_adj = d_mat.dot(adj_mat)
        norm_adj = norm_adj.dot(d_mat)
        norm_adj = norm_adj.tocsr()

        # norm_adj = norm_adj + γ
        '''下面计算γ'''
        c_mat = adj_mat.T.dot(adj_mat)
        atten_row_sum = np.array(c_mat.sum(axis=1))
        z_inv = np.power(atten_row_sum, -0.5).flatten()
        z_inv[np.isinf(z_inv)] = 0.
        z_mat = sp.diags(z_inv)
        cos_adj = z_mat.dot(adj_mat)
        cos_adj = cos_adj.dot(z_mat)
        cos_adj = cos_adj.tocsr()

        norm_cos_adj = norm_adj + cos_adj
        end = time()
        print(f"costing {end -start} s, saved norm_mat...")
        # sp.save_npz(self.save_path + '/s_pre_adj_mat.npz', norm_cos_adj)


        # self.Graph = self._convert_sp_mat_to_sp_tensor(norm_cos_adj).to(self.args.mdevice)
        # self.Graph = self.Graph.coalesce().to(self.args.mdevice)
        self.Graph = norm_cos_adj

        return self.Graph


# parser = argparse.ArgumentParser()
#
# parser.add_argument('--data_dir', type=str, default='../data/lastfm-20-100-400')
# # parser.add_argument('--data_dir', type=str, default='../data/movielens-1m-200-800')
# # parser.add_argument('--data_dir', type=str, default='../data/book_crossing-400-1600')
#
# parser.add_argument('--dataset_name', type=str, default='define')
# parser.add_argument('--seed', type=int, default=2022)
# parser.add_argument('--latent_dim_rec', type=int, default=64, help="the embedding size of lightGCN")
# parser.add_argument('--n_layer', type=int, default=3, help="the layer num of lightGCN")
# parser.add_argument('--keep_prob', type=float, default=0.6, help="the batch size for bpr loss training procedure")
# parser.add_argument('--pretrain', type=int, default=0, help='whether we use pretrained weight or not，0 is not use')
# parser.add_argument('--dropout', type=int, default=0, help="using the dropout or not，0 is not use")
# parser.add_argument('--epochs', type=int, default=500)
# parser.add_argument('--bpr_batch', type=int, default=1024, help="the batch size for bpr loss training procedure")
# parser.add_argument('--local_lr', type=float, default=0.001, help="the local model learning rate")
# parser.add_argument('--meta_lr', type=float, default=0.005, help="the meta model learning rate")
# parser.add_argument('--decay', type=float, default=1e-4, help="the weight decay for l2 normalization")
# parser.add_argument('--test_batch', type=int, default=100, help="the batch size of items for testing")
# parser.add_argument('--top_k', type=int, default=10, help="the size of recommendation")
# parser.add_argument('--embedding_dim', type=int, default=32)
# parser.add_argument('--first_fc_hidden_dim', type=int, default=64, help='Embedding dimension for item and user.')
# parser.add_argument('--second_fc_hidden_dim', type=int, default=32, help='Embedding dimension for item and user.')
#
# args = parser.parse_args()
# '''设置随机种子，使得生成的随机数固定不变'''
# torch.manual_seed(args.seed)
#
#
# '''设置运行的cuda'''
# GPU = torch.cuda.is_available()
# mdevice = torch.device('cuda:1' if GPU else 'cpu')
# args.mdevice = mdevice
# args.n_item = 3846  # lastfm-20
# args.m_user = 1872


# training_supp_path = args.data_dir + '/training/top_supp_35_pos.txt'
# train_dataset = '/training'
# training_supp_dataset = LoaderDataset(args, training_supp_path, train_dataset)
# print("-------------training_supp_dataset.getSparseGraph()-------------")
# print(training_supp_dataset.getSparseGraph())

# testing_supp_path = args.data_dir + '/testing/tail_supp_7_pos.txt'
# test_dataset = '/testing'
# testing_supp_dataset = LoaderDataset(args, testing_supp_path, test_dataset)
# print("-------------testing_supp_dataset.getSparseGraph()-------------")
# print(testing_supp_dataset.getSparseGraph())