
import random
import torch
from torch import nn, optim
import numpy as np
from time import time
from torch.autograd import Variable
from load_dataset import LoadSample
from utils import shuffle, minibatch, add_metric
from torch.nn import functional as F

'''对所有数据进行一次训练'''
def model_training(args, dataset, recommend_model):

    start = time()
    Recmodel = recommend_model
    Recmodel.train()    # 模型训练
    optimizer_params = {"lr": 1e-3, "weight_decay": 1e-5}
    optimizer = torch.optim.Adam(Recmodel.parameters(), **optimizer_params)

    '''加载和处理数据,得到S=[[item,user+,hist_user,label=1],[item,user-,hist_user,label=0] ...]'''
    S = LoadSample(dataset)

    items_id = torch.Tensor(S[:, 0]).long()
    users_id = torch.Tensor(S[:, 1]).long()
    history_users_id = torch.Tensor(S[:, 2:8]).long()
    label = torch.Tensor(S[:, 8]).long()

    items_id = items_id.to(args.mdevice)
    users_id = users_id.to(args.mdevice)
    history_users_id = history_users_id.to(args.mdevice)
    label = label.to(args.mdevice)

    '''utils.shuffle用于随机打乱顺序，但对应关系不变'''
    items_id, users_id, history_users_id, label = shuffle(items_id, users_id, history_users_id, label)

    num_batch = len(items_id) // args.batch_size + 1  # args.bpr_batch=2048
    total_loss = 0.
    '''将组合矩阵(items_id, pos_users_id, neg_items_id, history_users_id）分成batch_i批,进行分批训练'''
    for (batch_i, (batch_items, batch_users, batch_hist, batch_label)) in \
            enumerate(minibatch(items_id, users_id, history_users_id, label, batch_size=args.batch_size)):

        '''获取每个批次的loss'''
        pred_score = Recmodel(batch_items, batch_users, batch_hist)
        batch_loss = F.mse_loss(pred_score, batch_label.float().view(-1, 1))

        Recmodel.zero_grad()
        batch_loss.backward()
        optimizer.step()

        total_loss += batch_loss

    aver_loss = total_loss / num_batch
    end = time()

    return f"train_loss:{aver_loss:.3f} spend_times:{(end-start):.2f}"

'''对所有数据进行一次测试'''
def model_testing(args, dataset, recommend_model, item_size):
    Recmodel = recommend_model
    Recmodel.eval()    # 模型测试

    '''加载和处理数据'''
    S = LoadSample(dataset)

    items_id = torch.Tensor(S[:, 0]).long()
    users_id = torch.Tensor(S[:, 1]).long()
    history_users_id = torch.Tensor(S[:, 2:8]).long()
    label = torch.Tensor(S[:, 8]).long()

    items_id = items_id.to(args.mdevice)
    users_id = users_id.to(args.mdevice)
    history_users_id = history_users_id.to(args.mdevice)
    label = label.to(args.mdevice)
    test_data_len = len(dataset.trainUniqueItems)

    total_ndcg_list = []
    total_prec_list = []

    """这里的测试是按照一个item_id对应其所有交互的user_id来进行的，例如：
    37: [1240 483 149 2 658 246 95 539 628 1689 1800 53 47 179]
    会先预测item_id=37的user_id列表，然后对比其label，计算precision和ndcg"""
    for i in range(test_data_len):
        batch_items = items_id[i*item_size:(i+1)*item_size]
        batch_users = users_id[i*item_size:(i+1)*item_size]
        batch_hist_users = history_users_id[i*item_size:(i+1)*item_size]
        batch_labels = label[i*item_size:(i+1)*item_size]

        batch_items, batch_users, batch_hist_users, batch_labels = shuffle(batch_items, batch_users, batch_hist_users, batch_labels)

        pred_score = Recmodel(batch_items, batch_users, batch_hist_users)

        test_output_list, query_recom_list = pred_score.view(-1).sort(descending=True)
        test_prec, test_ndcg = add_metric(query_recom_list, batch_labels, topn=10)
        total_prec_list.append(test_prec)
        total_ndcg_list.append(test_ndcg)


    total_prec = sum(total_prec_list) / len(total_prec_list)
    total_ndcg = sum(total_ndcg_list) / len(total_ndcg_list)

    return total_prec, total_ndcg

