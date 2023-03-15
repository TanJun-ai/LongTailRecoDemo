

import random
import numpy as np
import math


'''对正负样本的抽样策略：均匀抽样,返回的是 np.array()类型，S = [[1269 3706 4989] ...]，
对movielens-1m数据集来说，support-set中的train的交互数量为497880，即S的大小为497880'''
def UniformSample(dataset):

    '''allPos为字典类型3947: array([   9,   10, ...])'''
    allPos = dataset.allPos  # 所有的正样本item_id

    '''抽样的结果为一个二维list，里面的每个一维list表示一个item对应的一个三元组<item, user+, user->
    S=[[ 864 1667 1334],[3940  380  318]...],这里抽样的长度为801371 interactions for training，
    因为训练集里面一共有801371条交互记录，需要对每条交互记录都输入到模型中训练，来学习模型的参数，
    在这些交互记录中，少数的热门item占据了大部分的交互记录，所以学出来的参数更偏向于热门的item，对热门item预测推荐就更准确'''

    item_num = dataset.n_item  # item的数量,这里是3948
    items = random.sample(range(0, dataset.n_item), item_num)  # 随机生成item_num个item_id，范围是（0, dataset.n_item）
    S = []  # 用来获取三元组<item, user+, user->

    '''这里循环了i=3948次'''

    for i, item in enumerate(items):

        posForItem = allPos[item]

        # 这个item没有对应的user_id，即item没有user的交互记录，则跳过这个item
        if len(posForItem) == 0:
            continue
        '''生成范围内不重复的随机整数，生成的个数为len(posForItem)'''
        posindex = random.sample(range(0, len(posForItem)), len(posForItem))

        for index in range(len(posForItem)):
            '''这里是从每个item对应的正样本user_id的list中随机抽取一个正样本posuser'''
            pos_index = posindex[index]
            posuser = posForItem[pos_index]

            '''负样本抽取策略：从0-m_user个user_id中随机找一个id出来，如果这个user_id在正样本list中，则继续找，
            直到找到的user_id不在正样本的list中，就是item对应的负样本'''
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

'''修改np.random.shuffle函数，使其适用于三个输入tensor，目的是打乱顺序，但对应关系的顺序不变
list1 = torch.tensor([20, 16, 10, 5])
list2 = torch.tensor([15, 10, 12, 9])
list3 = torch.tensor([25, 13, 19, 8])
===========list1, list2, list3 = shuffle(list1, list2, list3)===========
tensor([10,  5, 20, 16])
tensor([12,  9, 15, 10])
tensor([19,  8, 25, 13])
'''
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

    batch_size = kwargs.get('batch_size', 2048)

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


