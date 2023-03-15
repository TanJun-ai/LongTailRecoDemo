
import torch
import numpy as np

from scipy.sparse import csr_matrix
import scipy.sparse as sp

from time import time


class LoaderDataset:
    """
    Procceeding Dataset for movielens-1m ：这个数据集分成support-set 和 query-set，
    support-set 中的item最大值为3948，user最大值为6041， query-set 中的item最大值为3952，user最大值为6041
    """
    def __init__(self, parameter, train_path, save_path):

        self.trainDataSize = 0
        self.testDataSize = 0
        self.save_path = parameter.data_dir + save_path
        self.args = parameter
        trainUniqueItems, trainItem, trainUser = [], [], []
        self.n_item = 0
        self.m_user = 0
        with open(train_path) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
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
        '''因为数据的索引是从0开始的，这里要+1确保矩阵的大小能容纳item_id的最大值'''
        self.n_item = self.args.n_item
        self.m_user = self.args.m_user
        '''user 和正样本item 的交互矩阵，矩阵的元素值都为1.0，这里从0开始，即user_id为0的user其
        真实的user_id=1,(0, 1) 1.0。。。(6039, 3819)	1.0,数据类型为矩阵csr_matrix'''
        self.UserItemNet = csr_matrix((np.ones(len(self.trainItem)), (self.trainItem, self.trainUser)),
                                      shape=(self.n_item, self.m_user))

        self.users_D = np.array(self.UserItemNet.sum(axis=1)).squeeze()
        self.users_D[self.users_D == 0.] = 1
        self.items_D = np.array(self.UserItemNet.sum(axis=0)).squeeze()
        self.items_D[self.items_D == 0.] = 1.

        '''self.n_user=6040，拿到每个item对应的正样本user_id的list，self.allPos=[...,[...3671 3683 3703 3735 3751 3819]]'''
        self.allPos = self.getUserPosItems(list(range(self.n_item)))

        print(f"{parameter.dataset_name} is ready to go")

        self.Graph = None

    '''
    参数items:为list的item_id=[0,1,2,3...3947]，这里是0-3947,
    这里是将item 和 正样本user交互矩阵中的user_id取出来，即从。。。(3819, 6039)	1.0 中，
    取出self.UserItemNet[item].nonzero()[1]，即3819，组成一个2维list，
    [...,[...3671 3683 3703 3735 3751 3819]]，里面的每个1维list都是item对应的正样本user_id'''
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

    '''如果已经有了生成好的图数据，直接加载，如果没有，需要重新生成'''
    def getSparseGraph(self):
        print("loading adjacency matrix")

        # if self.Graph is None:
        try:
            pre_adj_mat = sp.load_npz(self.save_path + '/s_pre_adj_mat.npz')
            print("successfully loaded...")
            norm_adj = pre_adj_mat
        except :
            print("generating adjacency matrix")
            start = time()
            #  首先生成大小为(self.n_item + self.m_user, self.n_item + self.m_user)的矩阵
            adj_mat = sp.dok_matrix((self.n_item + self.m_user, self.n_item + self.m_user), dtype=np.float32)
            adj_mat = adj_mat.tolil()
            R = self.UserItemNet.tolil()
            adj_mat[:self.n_item, self.n_item:] = R
            adj_mat[self.n_item:, :self.n_item] = R.T
            adj_mat = adj_mat.todok()
            np.seterr(divide='ignore', invalid='ignore')  # 消除被除数为0的警告
            row_sum = np.array(adj_mat.sum(axis=1))
            d_inv = np.power(row_sum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat = sp.diags(d_inv)

            norm_adj = d_mat.dot(adj_mat)
            norm_adj = norm_adj.dot(d_mat)
            norm_adj = norm_adj.tocsr()
            end = time()
            print(f"costing {end -start} s, saved norm_mat...")
            sp.save_npz(self.save_path + '/s_pre_adj_mat.npz', norm_adj)

        self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
        self.Graph = self.Graph.coalesce().to(self.args.mdevice)

        return self.Graph




