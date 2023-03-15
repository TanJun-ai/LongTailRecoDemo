
import argparse
import os

import numpy as np
import random
import torch

import LightGCN
import training
from utils import add_metric
from load_dataset import LoaderDataset
from training import model_training

parser = argparse.ArgumentParser()
# parser.add_argument('--data_dir', type=str, default='../data/lastfm-20-100-400')
parser.add_argument('--data_dir', type=str, default='../data/movielens-1m-200-800')
parser.add_argument('--dataset_name', type=str, default='movielens-1m')
parser.add_argument('--seed', type=int, default=2022)
parser.add_argument('--latent_dim_rec', type=int, default=64, help="the embedding size of lightGCN")
parser.add_argument('--n_layer', type=int, default=3, help="the layer num of lightGCN")
parser.add_argument('--keep_prob', type=float, default=0.6, help="the batch size for bpr loss training procedure")
parser.add_argument('--pretrain', type=int, default=0, help='whether we use pretrained weight or not，0 is not use')
parser.add_argument('--dropout', type=int, default=0, help="using the dropout or not，0 is not use")
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--bpr_batch', type=int, default=2048, help="the batch size for bpr loss training procedure")
parser.add_argument('--local_lr', type=float, default=0.001, help="the local model learning rate")
parser.add_argument('--meta_lr', type=float, default=0.01, help="the meta model learning rate")
parser.add_argument('--decay', type=float, default=1e-4, help="the weight decay for l2 normalization")
parser.add_argument('--test_batch', type=int, default=100, help="the batch size of items for testing")
parser.add_argument('--top_k', type=int, default=10, help="the size of recommendation")
args = parser.parse_args()

'''设置随机种子，使得生成的随机数固定不变'''
def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed = args.seed
seed_everything(seed)

'''设置运行的cuda'''
GPU = torch.cuda.is_available()
mdevice = torch.device('cuda:0' if GPU else 'cpu')
args.mdevice = mdevice
# args.n_item = 3846
# args.m_user = 1872
# args.n_item = 3953
# args.m_user = 6041
args.n_item = 6041
args.m_user = 3953

print("==========================train-set-start=======================")
support_train_path = args.data_dir + '/testing/overall_test.txt'
support_dataset = LoaderDataset(args, support_train_path, '/testing')
print("support_train_items:" + str(support_dataset.n_item))
print("support_train_users:" + str(support_dataset.m_user))
print("support_train_len:" + str(len(support_dataset.trainUniqueItems)))
print("==========================train-set-end=========================")

print("==========================test-set-start=========================")
query_set = args.data_dir + '/testing/tail_query_14.txt'
query_set_label = args.data_dir + '/testing/tail_query_14_label.txt'
query_dataset = LoaderDataset(args, query_set, '/testing')
testing_query_size = len(query_dataset.trainUniqueItems)   # 测试集有1011条数据
test_items_id_list = []
test_users_id_list = []
test_labels_list = []
for i in range(testing_query_size):
    cache_list = []
    with open(query_set) as f:
        for l in f.readlines():
            if len(l) > 0:
                l = l.strip('\n').split(' ')
                cache_list.append(l)
    items_id = []
    for j in range(len(cache_list[i])-1):
        items_id.append(int(cache_list[i][0]))
    users_id = []
    users_id.extend(int(i) for i in cache_list[i][1:])
    test_items_id_list.append(torch.Tensor(items_id).long())
    test_users_id_list.append(torch.Tensor(users_id).long())

for j in range(testing_query_size):
    cache_list = []
    with open(query_set_label) as f:
        for l in f.readlines():
            if len(l) > 0:
                l = l.strip('\n').split(' ')
                cache_list.append(l)
    labels_id = []
    labels_id.extend(float(i) for i in cache_list[j][1:])
    test_labels_list.append(torch.Tensor(labels_id).long())

print("query_set_items:" + str(query_dataset.n_item))
print("query_set_users:" + str(query_dataset.m_user))
print("query_set_len:" + str(testing_query_size))
print("===========================test-set-end==========================")

Recmodel = LightGCN.BaseModel(args, support_dataset)
Recmodel = Recmodel.to(mdevice)
bpr = training.BPRLossTraining(Recmodel, args)
print("==========LightGCN-Model==========")
print(Recmodel)
print("==================================")


max_ndcg = 0.
max_prec = 0.

for epoch in range(args.epochs):

    '''使用BPR损失函数进行训练'''
    output_information = model_training(args, support_dataset, Recmodel, bpr)
    print(f'EPOCH[{epoch + 1}/{args.epochs}] {output_information}')

    '''对test-set数据集进行测试'''
    total_ndcg_list = []
    total_prec_list = []
    te_loss = []
    for j in range(testing_query_size):
        test_items_id_list[j] = test_items_id_list[j]
        test_users_id_list[j] = test_users_id_list[j]
        test_labels_list[j] = test_labels_list[j]
        ratings = Recmodel.getItemsRating(test_items_id_list[j], test_users_id_list[j])
        scores = (sum(ratings) / len(ratings)).view(-1, 1)

        output_list, y_recom_list = scores.view(-1).sort(descending=True)
        test_prec, test_ndcg = add_metric(y_recom_list, test_labels_list[j], topn=10)
        total_prec_list.append(test_prec)
        total_ndcg_list.append(test_ndcg)

    total_prec = sum(total_prec_list) / len(total_prec_list)
    total_ndcg = sum(total_ndcg_list) / len(total_ndcg_list)
    if total_prec > max_prec:
        max_prec = total_prec
    if total_ndcg.item() > max_ndcg:
        max_ndcg = total_ndcg.item()
    '''每循环训练10次，就打印结果'''
    if epoch % 10 == 0:
        print("----------------------query-testing---------------------")
        print("TOP-10: query_prec:{:.4f}\t\tquery_ndcg:{:.4f}".format(total_prec, total_ndcg.item()))
        print("max_prec:{:.4f}\t\tmax_ndcg:{:.4f}".format(max_prec, max_ndcg))
        print("-------------------------------------------------------")




