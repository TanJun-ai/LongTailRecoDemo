
import os
from time import time
import numpy as np
import random

import argparse
import torch
import json
from torch.autograd import Variable
from Config import config_ml as config
from HeteML_new import HML

np.random.seed(2022)
torch.manual_seed(2022)

GPU = torch.cuda.is_available()
mdevice = torch.device('cuda:1' if GPU else 'cpu')

'''将int型转成one-hot向量'''
def to_onehot_dict(list):
    dict={}
    length = len(list)
    for index, element in enumerate(list):
        vector = torch.zeros(1, length).long()
        element = int(element)
        vector[:, element] = 1.0
        dict[element] = vector
    return dict

# data_dir = '../data/lastfm-20-100-400'
# item_dim = 3846
# user_dim = 1872
data_dir = '../data/movielens-1m-200-800'
item_dim = 3953
user_dim = 6041
# data_dir = '../data/book_crossing-400-1600'
# item_dim = 8000
# user_dim = 2947

start = time()
print("=======================training-set-start=========================")
with open('{}/item_list.json'.format(data_dir), 'r', encoding='utf-8') as f:
    itemids = json.loads(f.read())
with open('{}/user_list.json'.format(data_dir), 'r', encoding='utf-8') as g:
    userids = json.loads(g.read())
item_dict = to_onehot_dict(itemids)
user_dict = to_onehot_dict(userids)

top_supp_path = data_dir + '/training/top_supp_70.txt'
top_supp_label_path = data_dir + '/training/top_supp_70_label.txt'
top_query_path = data_dir + '/training/top_query_14.txt'
top_query_label_path = data_dir + '/training/top_query_14_label.txt'

training_set_size = 200

supp_xs_s = []
supp_ys_s = []
supp_um = []
query_xs_s = []
query_ys_s = []
query_um = []

for idx in range(training_set_size):
    supp_x_cache = []  # 暂时缓存的每一列U-I总数据
    support_x_total = None  # 将100条user_id+item_id串联起来
    with open(top_supp_path) as fo:
        for lt in fo.readlines():
            if len(lt) > 0:
                lt = lt.strip().strip('\n').split(' ')
                supp_x_cache.append(lt)

    supp_um_one = []
    temp_um = []
    for j in range(len(supp_x_cache[idx]) - 1):
        item_id = int(supp_x_cache[idx][0])
        user_id = int(supp_x_cache[idx][j + 1])

        temp_um.append(user_dict[user_id][0].tolist())  # 新增user-item路径数据
        temp_x_tensoe = torch.cat((user_dict[user_id], item_dict[item_id]), 1)

        try:
            support_x_total = torch.cat((support_x_total, temp_x_tensoe), 0)
        except:
            support_x_total = temp_x_tensoe
    supp_xs_s.append(torch.tensor(support_x_total))
    for i in range(len(supp_x_cache[idx]) - 1):
        supp_um_one.append(torch.tensor(temp_um).float())
    supp_um.append(supp_um_one)

    supp_y_cache = []
    with open(top_supp_label_path) as su:
        for lu in su.readlines():
            if len(lu) > 0:
                lu = lu.strip().strip('\n').split(' ')
                supp_y_cache.append(lu)
    support_y_app = torch.FloatTensor([float(x) for x in supp_y_cache[idx][1:]])
    supp_ys_s.append(support_y_app)

    query_x_cache = []
    query_x_total = None
    with open(top_query_path) as qu:
        for lq in qu.readlines():
            if len(lq) > 0:
                lq = lq.strip().strip('\n').split(' ')
                query_x_cache.append(lq)

    query_um_one = []
    query_temp_um = []
    for n in range(len(query_x_cache[idx]) - 1):
        item_id_q = int(query_x_cache[idx][0])
        user_id_q = int(query_x_cache[idx][n + 1])
        query_temp_um.append(user_dict[user_id_q][0].tolist())
        query_x_tensor = torch.cat((user_dict[user_id_q], item_dict[item_id_q]), 1)
        try:
            query_x_total = torch.cat((query_x_total, query_x_tensor), 0)
        except:
            query_x_total = query_x_tensor
    query_xs_s.append(query_x_total)
    for i in range(len(query_x_cache[idx]) - 1):
        query_um_one.append(torch.tensor(query_temp_um).float())
    query_um.append(query_um_one)

    query_y_cache = []
    with open(top_query_label_path) as qe:
        for le in qe.readlines():
            if len(le) > 0:
                le = le.strip().strip('\n').split(' ')
                query_y_cache.append(le)
    query_y_app = torch.FloatTensor([float(x) for x in query_y_cache[idx][1:]])
    query_ys_s.append(query_y_app)

train_dataset = list(zip(supp_xs_s, supp_ys_s, supp_um, query_xs_s, query_ys_s, query_um))
del (supp_xs_s, supp_ys_s, supp_um, query_xs_s, query_ys_s, query_um)
print("item-nums:" + str(item_dim))
print("user-nums:" + str(user_dim))
print("training_set_size:" + str(training_set_size))
print("=========================training-set-end==========================")


print("=========================testing-set-start=========================")
tail_supp_path = data_dir + '/testing/tail_supp_14.txt'
tail_supp_label_path = data_dir + '/testing/tail_supp_14_label.txt'
tail_query_path = data_dir + '/testing/tail_query_14.txt'
tail_query_label_path = data_dir + '/testing/tail_query_14_label.txt'

testing_set_size = 800

supp_xs_s = []
supp_ys_s = []
supp_um = []
query_xs_s = []
query_ys_s = []
query_um = []

for idxx in range(testing_set_size):

    '''supp_x,supp_um'''
    supp_x_cache = []  # 暂时缓存的每一列U-I总数据
    support_x_total = None  # 将100条user_id+item_id串联起来
    with open(tail_supp_path) as fo:
        for lt in fo.readlines():
            if len(lt) > 0:
                lt = lt.strip().strip('\n').split(' ')
                supp_x_cache.append(lt)

    supp_um_one = []
    temp_um = []
    for j in range(len(supp_x_cache[idxx]) - 1):
        item_id = int(supp_x_cache[idxx][0])
        user_id = int(supp_x_cache[idxx][j + 1])

        temp_um.append(user_dict[user_id][0].tolist())
        temp_x_tensoe = torch.cat((user_dict[user_id], item_dict[item_id]), 1)
        try:
            support_x_total = torch.cat((support_x_total, temp_x_tensoe), 0)
        except:
            support_x_total = temp_x_tensoe
    supp_xs_s.append(support_x_total)
    for i in range(len(supp_x_cache[idxx]) - 1):
        supp_um_one.append(torch.tensor(temp_um).float())
    supp_um.append(supp_um_one)

    '''label'''
    supp_y_cache = []
    with open(tail_supp_label_path) as su:
        for lu in su.readlines():
            if len(lu) > 0:
                lu = lu.strip().strip('\n').split(' ')
                supp_y_cache.append(lu)
    support_y_app = torch.FloatTensor([float(x) for x in supp_y_cache[idxx][1:]])
    supp_ys_s.append(support_y_app)

    '''query_x,query_um'''
    query_x_cache = []
    query_x_total = None
    with open(tail_query_path) as qu:
        for lq in qu.readlines():
            if len(lq) > 0:
                lq = lq.strip().strip('\n').split(' ')
                query_x_cache.append(lq)

    query_um_one = []
    query_temp_um = []
    for n in range(len(query_x_cache[idxx]) - 1):
        item_id_q = int(query_x_cache[idxx][0])
        user_id_q = int(query_x_cache[idxx][n + 1])

        query_temp_um.append(user_dict[user_id_q][0].tolist())
        query_x_tensor = torch.cat((user_dict[user_id_q], item_dict[item_id_q]), 1)
        try:
            query_x_total = torch.cat((query_x_total, query_x_tensor), 0)
        except:
            query_x_total = query_x_tensor
    query_xs_s.append(query_x_total)
    for i in range(len(query_x_cache[idxx]) - 1):
        query_um_one.append(torch.tensor(query_temp_um).float())
    query_um.append(query_um_one)

    '''label'''
    query_y_cache = []
    with open(tail_query_label_path) as qe:
        for le in qe.readlines():
            if len(le) > 0:
                le = le.strip().strip('\n').split(' ')
                query_y_cache.append(le)
    query_y_app = torch.FloatTensor([float(x) for x in query_y_cache[idxx][1:]])
    query_ys_s.append(query_y_app)

    # item_x = Variable(supp_xs_s[idxx][:, 1872:], requires_grad=False).float()
    # user_x = Variable(supp_xs_s[idxx][:, 0:1872], requires_grad=False).float()
    # item_id_list = np.argmax(item_x.tolist(), axis=1)
    # user_id_list = np.argmax(user_x.tolist(), axis=1)
    # print("-------item_id_list-------")
    # print(item_id_list)
    # print("-------user_id_list-------")
    # print(user_id_list)

test_dataset = list(zip(supp_xs_s, supp_ys_s, supp_um, query_xs_s, query_ys_s, query_um))
del (supp_xs_s, supp_ys_s, supp_um, query_xs_s, query_ys_s, query_um)
print("testing_set_size:" + str(testing_set_size))
print("========================testing-set-end==========================")
end = time()
print(f'Load Spend Time: [{end - start:.3f}]')


def training_and_testing(model, device):

    print('training model...')
    model.to(device)
    model.train()

    batch_size = config['batch_size']
    num_epoch = config['num_epoch']

    max_prec10 = 0.
    max_ndcg10 = 0.
    max_prec8 = 0.
    max_ndcg8 = 0.
    max_prec5 = 0.
    max_ndcg5= 0.

    for epoch in range(num_epoch):  # 20

        loss = []
        start_ = time()

        random.shuffle(train_dataset)
        num_batch = int(len(train_dataset) / batch_size)  # len(train_dataset)=100 batch_size=32
        supp_xs_s, supp_ys_s, supp_mps_s, query_xs_s, query_ys_s, query_mps_s = zip(*train_dataset)  # supp_um_s:(list,list,...,2553)

        for i in range(num_batch):  # each batch contains some tasks (each task contains a support set and a user_cold_testing set)
            support_xs = list(supp_xs_s[batch_size * i:batch_size * (i + 1)])
            support_ys = list(supp_ys_s[batch_size * i:batch_size * (i + 1)])
            support_mps = list(supp_mps_s[batch_size * i:batch_size * (i + 1)])
            query_xs = list(query_xs_s[batch_size * i:batch_size * (i + 1)])
            query_ys = list(query_ys_s[batch_size * i:batch_size * (i + 1)])
            query_mps = list(query_mps_s[batch_size * i:batch_size * (i + 1)])

            _loss, _prec10, _ndcg10, _prec8, _ndcg8, _prec8, _ndcg8 = model.global_update(
                support_xs, support_ys, support_mps, query_xs, query_ys, query_mps, device)
            loss.append(_loss)

        print('epoch: {}, loss: {:.4f}, cost time: {:.1f}s'.format(epoch, sum(loss)/len(loss), time() - start_))

        total_prec10, total_ndcg10, total_prec8, total_ndcg8, total_prec5, total_ndcg5 = testing(model, device)
        model.train()
        if total_prec10 > max_prec10:
            max_prec10 = total_prec10
        if total_ndcg10.item() > max_ndcg10:
            max_ndcg10 = total_ndcg10.item()

        if total_prec8 > max_prec8:
            max_prec8 = total_prec8
        if total_ndcg8.item() > max_ndcg8:
            max_ndcg8 = total_ndcg8.item()

        if total_prec5 > max_prec5:
            max_prec5 = total_prec5
        if total_ndcg5.item() > max_ndcg5:
            max_ndcg5 = total_ndcg5.item()
        if epoch % 10 == 0:
            print("----------------------query-testing---------------------")
            print("TOP-10: query_prec:{:.4f}\t\tquery_ndcg:{:.4f}".format(total_prec10, total_ndcg10.item()))
            print("TOP-8: query_prec:{:.4f}\t\tquery_ndcg:{:.4f}".format(total_prec8, total_ndcg8.item()))
            print("TOP-5: query_prec:{:.4f}\t\tquery_ndcg:{:.4f}".format(total_prec5, total_ndcg5.item()))
            print("TOP-10:max_prec:{:.4f}\t\tmax_ndcg:{:.4f}".format(max_prec10, max_ndcg10))
            print("TOP-8:max_prec:{:.4f}\t\tmax_ndcg:{:.4f}".format(max_prec8, max_ndcg8))
            print("TOP-5:max_prec:{:.4f}\t\tmax_ndcg:{:.4f}".format(max_prec5, max_ndcg5))
            print("-------------------------------------------------------")

def testing(model, device):
    model.to(device)
    model.eval()
    total_prec10, total_ndcg10, total_prec8, total_ndcg8, total_prec5, total_ndcg5 = evaluate(model, device)
    return total_prec10, total_ndcg10, total_prec8, total_ndcg8, total_prec5, total_ndcg5

def evaluate(model, device):

    supp_xs_s, supp_ys_s, supp_mps_s, query_xs_s, query_ys_s, query_mps_s = zip(*test_dataset)  # supp_um_s:(list,list,...,2553)

    total_prec_list10 = []
    total_ndcg_list10 = []
    total_prec_list8 = []
    total_ndcg_list8= []
    total_prec_list5= []
    total_ndcg_list5 = []

    for i in range(len(test_dataset)):  # each task
        _loss, _test_prec10, _test_ndcg10, _test_prec8, _test_ndcg8, _test_prec5, _test_ndcg5 = model.evaluation(
            supp_xs_s[i], supp_ys_s[i], supp_mps_s[i], query_xs_s[i], query_ys_s[i], query_mps_s[i], device)

        total_prec_list10.append(_test_prec10)
        total_ndcg_list10.append(_test_ndcg10)
        total_prec_list8.append(_test_prec8)
        total_ndcg_list8.append(_test_ndcg8)
        total_prec_list5.append(_test_prec5)
        total_ndcg_list5.append(_test_ndcg5)

    total_prec10 = sum(total_prec_list10) / len(total_prec_list10)
    total_ndcg10 = sum(total_ndcg_list10) / len(total_ndcg_list10)
    total_prec8 = sum(total_prec_list8) / len(total_prec_list8)
    total_ndcg8 = sum(total_ndcg_list8) / len(total_ndcg_list8)
    total_prec5 = sum(total_prec_list5) / len(total_prec_list5)
    total_ndcg5 = sum(total_ndcg_list5) / len(total_ndcg_list5)

    return total_prec10, total_ndcg10, total_prec8, total_ndcg8, total_prec5, total_ndcg5


if __name__ == "__main__":

    print("========================config=========================")
    print(config)

    # training model.
    model_name = 'mp_update'
    hml = HML(config, model_name, device=mdevice)
    print("=======================model_name=====================")
    print(model_name)

    print("=======================training-start=====================")
    training_and_testing(hml, device=mdevice)
    print("=======================training-end=====================")

    print("=======================testing-start=====================")
    testing(hml, device=mdevice)
    print("=======================testing-end=====================")
