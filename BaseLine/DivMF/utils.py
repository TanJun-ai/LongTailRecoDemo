
'''
Top-K Diversity Regularizer
This software may be used only for research evaluation purposes.
For other purposes (e.g., commercial), please contact the authors.
'''

import random
from torch.utils.data import DataLoader
import numpy as np
from numpy.core.numeric import indices
import pandas as pd
import scipy.sparse as sp
from scipy.sparse import csr_matrix
import torch
from torch._C import Value
import torch.utils.data as data
from tqdm import tqdm
import math

CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if CUDA else 'cpu')

def NDCG(ranked_list, ground_truth, topn):
    dcg = 0
    idcg = IDCG(ground_truth, topn)

    for i in range(topn):
        id = ranked_list[i]
        dcg += ((2 ** ground_truth[id]) - 1)/math.log(i+2, 2)

    return dcg / idcg

def IDCG(ground_truth, topn):
    t = [a for a in ground_truth]
    t.sort(reverse=True)
    idcg = 0
    for i in range(topn):
        idcg += ((2**t[i]) - 1) / math.log(i+2, 2)
    return idcg

def precision(ranked_list, ground_truth, topn):
    t = [a for a in ground_truth]
    t.sort(reverse=True)
    t = t[:topn]
    hits = 0
    for i in range(topn):
        id = ranked_list[i]
        if ground_truth[id] in t:
            t.remove(ground_truth[id])
            hits += 1
    pre = hits/topn
    return pre

def add_metric(recommend_list, all_group_list, topn):

    ndcg = NDCG(recommend_list, all_group_list, topn)
    prec = precision(recommend_list, all_group_list, topn)

    return prec, ndcg

def set_seed(seed):
    '''
    Set pytorch random seed as seed.
    '''
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if CUDA:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if use multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def make_rec_list(model, top_k, user_num, item_num, train_data, device=DEVICE):
    '''
    Build recommendation lists from the model
    model : recommendation model
    top_k : length of a recommendation list
    user_num : number of users in dataset
    item_num : number of items in dataset
    train_data : lists of items that a user interacted in training dataset
    device : device where the model mounted on
    '''
    rtn = []
    for u in range(user_num):
        '''获取train_data中没有的item_id，即负样本，对这些负样本输入到model里面进行打分，
        然后按照得分高低排序，找出前10个分数最高的item_id，即为预测的列表'''
        items = torch.tensor(list(set(range(item_num))-set(train_data[u]))).to(device)
        u = torch.tensor([u]).to(device)
        score, _ = model(u, items, items)
        _, indices = torch.topk(score, top_k)
        recommends = torch.take(items, indices).cpu().numpy().tolist()
        rtn.append(recommends)
    return rtn

def load_all(train_path, test_neg):
    """ We load all the three file here to save time in each epoch. """
    '''
    Load dataset from given path
    trn_path : path of training dataset
    test_neg : path of test dataset
    '''
    train_data = pd.read_csv(
        train_path,
        sep='\t', header=None, names=['user', 'item'],
        usecols=[0, 1], dtype={0: np.int32, 1: np.int32})

    user_num = train_data['user'].max() + 1
    item_num = train_data['item'].max() + 1

    train_data = train_data.values.tolist()
    # print("----------train_data-----------")
    # print(train_data)

    # load ratings as a dok matrix
    train_mat = sp.dok_matrix((user_num, item_num), dtype=np.float32)
    for x in train_data:
        train_mat[x[0], x[1]] = 1.0

    test_data = []
    with open(test_neg, 'r') as fd:
        line = fd.readline()
        while line != None and line != '':
            arr = line.split('\t')
            u = eval(arr[0])[0]
            test_data.append([u, eval(arr[0])[1]])
            for i in arr[1:]:
                test_data.append([u, int(i)])
            line = fd.readline()
    return train_data, test_data, user_num, item_num, train_mat


'''训练集的正负样本策略：1个正样本对应4个负样本；[u, i, j] = [user, item+, item-]，
所以如果train-set有60000个[user,item+]，则经过BPRData的ng_sample方法后，会产生60000x4个[user, item+, item-]'''
class BPRData(data.Dataset):
    def __init__(self, features, num_item, train_mat=None, num_ng=0, is_training=None):
        super(BPRData, self).__init__()
        """ Note that the labels are only useful when training, we thus 
            add them in the ng_sample() function.
        features : data
        num_item : number of items
        train_mat : interaction matrix
        num_neg : number of negative samples
        is_training : is model training
        """
        self.features = features
        self.num_item = num_item
        self.train_mat = train_mat
        self.num_ng = num_ng
        self.is_training = is_training

    def ng_sample(self):
        '''
        Sample negative items for MFBPR
        '''
        assert self.is_training, 'no need to sampling when testing'

        self.features_fill = []
        n = 0
        for x in self.features:
            n += 1
            print(n)

            u, i = x[0], x[1]
            for t in range(self.num_ng):
                j = np.random.randint(self.num_item)
                while (u, j) in self.train_mat:
                    j = np.random.randint(self.num_item)
                self.features_fill.append([u, i, j])


    def __len__(self):
        '''
        Number of instances.
        '''
        return self.num_ng * len(self.features) if \
            self.is_training else len(self.features)

    def __getitem__(self, idx):
        '''
        Grab an instance.
        '''
        features = self.features_fill if \
            self.is_training else self.features

        user = features[idx][0]
        item_i = features[idx][1]
        item_j = features[idx][2] if \
            self.is_training else features[idx][1]
        return user, item_i, item_j


'''movielens-1m的user_num=6040，item_num=3952，train_mat为训练集的 user-[item1,item2,...]交互矩阵,
test_data是1个正样本(第一个)对应99个负样本(留一法)，格式：[[6039, 434], [6039, 3289], ...]
train_data是训练集的[user-item]正样本集，格式：[[0, 32], [0, 4],...]
'''
# train_data, test_data, user_num, item_num, train_mat = load_all(train_path, test_path)
# print("=======len-train-data=======")
# print(len(train_data))
# train_dataset = BPRData(train_data, item_num, train_mat, 4, True)
# train_loader = DataLoader(train_dataset, batch_size=4096, shuffle=True, num_workers=0)
# train_loader.dataset.ng_sample()    # 生成正负样本BPR对[u, i, j] = [user, item+, item-]
'''这里一共有60000x4个[user, item+, item-]，经过DataLoader函数的分批操作，得到
60000x4/4096=59批，所以下面要循环59次才可以将train-set的数据全部取出来；
user:4096,tensor([410, 290, 116,  ..., 273, 354, 284])
item_i（正样本）:4096,tensor([1319,   26,  860,  ...,  919,  618,  174])
item_j（负样本）：4096，tensor([1636, 2095, 1370,  ...,  561, 1637, 2519])'''

'''移到cpu上进行优化，在这里没有用，本来就是用cpu跑的'''
def optimizer_to(optim, device):
    '''
    Move optimizer to target device
    optim : optimizer
    device : target device
    '''
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


class LoaderDataset:
    """
    Procceeding Dataset for lastfm20 ：这个数据集分成support-set 和 query-set，
    support-set 中的item最大值为3846，user最大值为1872， query-set 中的item最大值为3846，user最大值为1872
    """
    def __init__(self, config, tra_path):
        self.n_item = 0
        self.m_user = 0
        self.trainDataSize = 0
        self.testDataSize = 0

        trainUniqueItems, trainItem, trainUser = [], [], []

        with open(tra_path) as f:
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
        self.n_item = config['n_item']
        self.m_user = config['m_user']
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

    def getUserPosItems(self, items):
        posUsers = {}
        for item in items:
            posUsers[item] = self.UserItemNet[item].nonzero()[1]

        return posUsers


def UniformSample(dataset, neg_num):

    '''allPos是每个item_id对应的正样本user_ids，为字典类型 3947: array([9, 10, ...])'''
    allPos = dataset.allPos  # 所有的正样本item_id

    item_num = dataset.n_item  # item的数量,这里是3948
    items = random.sample(range(0, dataset.n_item), item_num)  # 随机生成item_num个item_id，范围是（0, dataset.n_item）
    S = []  # 用来获取三元组<item, user+, user->

    for i, item in enumerate(items):

        posForItem = allPos[item]
        # 这个item没有对应的user_id，即item没有user的交互记录，则跳过这个item
        if len(posForItem) == 0:
            continue
        '''生成范围内不重复的随机整数，生成的个数为len(posForItem)'''
        posindex = random.sample(range(0, len(posForItem)), len(posForItem))
        '''
        item和它对应的正样本user(item:posForItem)
        1088:[53  155  156  196  419  422  510  514  756  795  846 1213 1372 1498 1528 1545 1584 1675 1691 2580]'''
        for index in range(len(posForItem)):
            '''这里是从每个item对应的正样本user_id的list中随机抽取一个正样本posuser'''
            pos_index = posindex[index]
            posuser = posForItem[pos_index]  # 从posForItem抽取一个正样本
            '''负样本抽取策略：从0-m_user个user_id中随机找一个id出来，如果这个user_id在正样本list中，则继续找，
            直到找到的user_id不在正样本的list中，就是item对应的负样本'''
            for num in range(neg_num):
                while True:
                    neguser = np.random.randint(0, dataset.m_user)
                    if neguser in posForItem:
                        continue
                    else:
                        break
                "返回三元组<item, user+, user->，len(posForItem)为每个item对应的交互user_id的集合"
                S.append([item, posuser, neguser])

    Sample = np.array(S)

    return Sample


def shuffle(*arrays, **kwargs):

    require_indices = kwargs.get('indices', False)

    if len(set(len(x) for x in arrays)) != 1:
        raise ValueError('All inputs to shuffle must have '
                         'the same length.')

    shuffle_indices = np.arange(len(arrays[0]))
    np.random.shuffle(shuffle_indices)

    if len(arrays) == 1:
        result = arrays[0][shuffle_indices]
    else:
        result = tuple(x[shuffle_indices] for x in arrays)

    if require_indices:
        return result, shuffle_indices
    else:
        return result

def minibatch(*tensors, **kwargs):

    batch_size = kwargs.get('batch_size', 1028)

    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)