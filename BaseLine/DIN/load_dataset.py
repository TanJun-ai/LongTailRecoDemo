
import torch
import random
import numpy as np
import argparse
from scipy.sparse import csr_matrix
import scipy.sparse as sp
from time import time

'''加载路径数据集，数据集格式：item:[user1,user2,...]'''
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

'''处理数据集，获得[item,user+,user-,history_inter]，长度为交互记录的长度，这里是2800，即有2800个一维list
[[1231, 1616, 358, 1180, 1329, 1333, 1375, 1502, 1531], [1231, 1333, 238, 1180, 1329, 1375, 1502, 1531, 1616],...]'''
def LoadSample(dataset):
    """allPos是每个item_id对应的正样本user_ids，为字典类型 3947: array([9, 10, ...])"""

    allPos = dataset.allPos  # 所有的正样本item_id

    item_num = dataset.n_item  # item的数量,这里是3948
    items = random.sample(range(0, dataset.n_item), item_num)  # 随机生成item_num个item_id，范围是（0, dataset.n_item）
    S = []  # 用来获取三元组<item, user+, history_inter, label=1>,<item, user-, history_inter, label=0>

    for i, item in enumerate(items):

        posForItem = allPos[item]
        # 这个item没有对应的user_id，即item没有user的交互记录，则跳过这个item,
        # len(posForItem)为每个item对应的交互user_id的集合
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
            history_items = list(posForItem)  # history_items是抽取一个正样本后，剩下的item_id都是这个正样本的历史交互记录
            history_items.remove(posuser)
            '''np.array是具有相同长度的数组，所以要取最小数量的那个作为history_items，否则会报错误：
            IndexError: too many indices for array: array is 1-dimensional, but 2 were indexed'''
            history_items = history_items[0:6]      # 取前6个hist_user_id即可

            '''负样本抽取策略：从0-m_user个user_id中随机找一个id出来，如果这个user_id在正样本list中，则继续找，
            直到找到的user_id不在正样本的list中，就是item对应的负样本'''
            while True:
                neguser = np.random.randint(0, dataset.m_user)
                if neguser in posForItem:
                    continue
                else:
                    break
            """返回正样本[item, user+, hist_user,label+]，负样本[item, user-, hist_user,label-]
            正样本的label为int型的1，负样本的为0，一条记录的长度为9，item(1), user+(1), hist_user(6), label+(1)
            """
            S.append([item, posuser] + history_items + [1])
            S.append([item, neguser] + history_items + [0])

    Sample = np.array(S)

    return Sample
