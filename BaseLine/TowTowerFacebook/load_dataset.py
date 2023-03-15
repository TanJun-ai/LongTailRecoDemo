

import random
import numpy as np
from scipy.sparse import csr_matrix


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

        print(f"{self.trainDataSize} interactions for dataset")

        self.n_item = self.args.n_item
        self.m_user = self.args.m_user
        self.UserItemNet = csr_matrix(
	        (np.ones(len(self.trainItem)), (self.trainItem, self.trainUser)), shape=(self.n_item, self.m_user))
        self.allPos = self.getUserPosItems(list(range(self.n_item)))

    def getUserPosItems(self, items):
        posUsers = {}
        for item in items:
            posUsers[item] = self.UserItemNet[item].nonzero()[1]
        return posUsers


def LoadSample(dataset, data_type):
    """allPos是每个item_id对应的正样本user_ids，为字典类型 3947: array([9, 10, ...])"""

    allPos = dataset.allPos  # 所有的正样本item_id

    item_num = dataset.n_item
    items = random.sample(range(0, dataset.n_item), item_num)
    S = []
    for i, item in enumerate(items):
        posForItem = allPos[item]
        if len(posForItem) == 0:
            continue
        '''生成范围内不重复的随机整数，生成的个数为len(posForItem)'''
        posindex = random.sample(range(0, len(posForItem)), len(posForItem))

        for index in range(len(posForItem)):
            '''这里是从每个item对应的正样本user_id的list中随机抽取一个正样本posuser'''
            pos_index = posindex[index]
            posuser = posForItem[pos_index]  # 从posForItem抽取一个正样本

            '''负样本抽取策略'''
            while True:
                neguser = np.random.randint(0, dataset.m_user)
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


