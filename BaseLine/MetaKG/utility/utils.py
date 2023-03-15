
import numpy as np
import math
import random
from scipy.sparse import csr_matrix

def shuffle(*arrays, **kwargs):

    require_indices = kwargs.get('indices', False)

    if len(set(len(x) for x in arrays)) != 1:
        raise ValueError('All inputs to shuffle must have the same length.')

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

    batch_size = kwargs.get('batch_size', 1024)

    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)

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


class LoaderDataset:
    """
    Procceeding Dataset for lastfm20 ：这个数据集分成support-set 和 query-set，
    support-set 中的item最大值为3846，user最大值为1872， query-set 中的item最大值为3846，user最大值为1872
    """
    def __init__(self, parameter, train_path):
        self.n_item = 0
        self.m_user = 0
        self.trainDataSize = 0
        self.testDataSize = 0

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
        self.UserItemNet = csr_matrix((np.ones(len(self.trainItem)),
                                       (self.trainItem, self.trainUser)), shape=(self.n_item, self.m_user))
        '''self.n_user=6040，拿到每个item对应的正样本user_id的list，self.allPos=[...,[...3671 3683 3703 3735 3751 3819]]'''
        self.allPos = self.getUserPosItems(list(range(self.n_item)))

    def getUserPosItems(self, items):
        posUsers = {}
        for item in items:
            posUsers[item] = self.UserItemNet[item].nonzero()[1]
        return posUsers

def LoadSample(dataset, data_type):
    """allPos是每个item_id对应的正样本user_ids，为字典类型 3947: array([9, 10, ...])"""

    allPos = dataset.allPos  # 所有的正样本item_id

    item_num = dataset.n_item  # item的数量,这里是3948
    items = random.sample(range(0, dataset.n_item), item_num)  # 随机生成item_num个item_id，范围是（0, dataset.n_item）
    S = []  # 用来获取三元组<item, user+, history_inter, label=1>,<item, user-, history_inter, label=0>

    for i, item in enumerate(items):
        posForItem = allPos[item]
        if len(posForItem) == 0:
            continue
        '''生成范围内不重复的随机整数，生成的个数为len(posForItem)'''
        posindex = random.sample(range(0, len(posForItem)), len(posForItem))
        '''
        item和它对应的正样本user(item:posForItem)
        1088:[53  155  156  196  419  422  510  514  756  795  846 1213 1372 1498 1528 1545 1584 1675 1691 2580]
        '''
        for index in range(len(posForItem)):
            '''这里是从每个item对应的正样本user_id的list中随机抽取一个正样本posuser'''
            pos_index = posindex[index]
            posuser = posForItem[pos_index]  # 从posForItem抽取一个正样本

            '''负样本抽取策略：从0-m_user个user_id中随机找一个id出来，如果这个user_id在正样本list中，则继续找，
            直到找到的user_id不在正样本的list中，就是item对应的负样本'''
            while True:
                # neguser = np.random.randint(0, dataset.m_user)
                neguser = np.random.randint(0, 250)
                if neguser in posForItem:
                    continue
                else:
                    break
            if data_type == "train":
                """
                返回正负样本[item, user+, user-]
                """
                S.append([item, posuser, neguser])
            else:
                S.append([item, posuser] + [1])
                S.append([item, neguser] + [0])

    Sample = np.array(S)

    return Sample