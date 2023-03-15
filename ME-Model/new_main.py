
import argparse
import os
import pickle
from time import time

from torch.nn import functional as F
import numpy as np
import random
import torch
from torch.autograd import Variable
from model import BaseModel, BPRLossTraining
from dataset_load import LoaderDataset
from utils import add_metric
from metaga_training import model_training, supp_testing, query_testing

torch.set_printoptions(threshold=np.inf)

parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', type=str, default='../data/lastfm-20-100-400')
# parser.add_argument('--data_dir', type=str, default='../data/movielens-1m-200-800')
# parser.add_argument('--data_dir', type=str, default='../data/book_crossing-400-1600')
# parser.add_argument('--data_dir', type=str, default='../data/book_crossing_100_400')

parser.add_argument('--dataset_name', type=str, default='define')
parser.add_argument('--seed', type=int, default=2022)
parser.add_argument('--latent_dim_rec', type=int, default=64, help="the embedding size of lightGCN")
parser.add_argument('--n_layer', type=int, default=3, help="the layer num of lightGCN")
parser.add_argument('--keep_prob', type=float, default=0.6, help="the batch size for bpr loss training procedure")
parser.add_argument('--pretrain', type=int, default=0, help='whether we use pretrained weight or not，0 is not use')
parser.add_argument('--dropout', type=int, default=0, help="using the dropout or not，0 is not use")
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--bpr_batch', type=int, default=1024, help="the batch size for bpr loss training procedure")
parser.add_argument('--local_lr', type=float, default=0.001, help="the local model learning rate")
parser.add_argument('--meta_lr', type=float, default=0.005, help="the meta model learning rate")
parser.add_argument('--decay', type=float, default=1e-4, help="the weight decay for l2 normalization")
parser.add_argument('--test_batch', type=int, default=100, help="the batch size of items for testing")
parser.add_argument('--top_k', type=int, default=10, help="the size of recommendation")
parser.add_argument('--embedding_dim', type=int, default=32)
parser.add_argument('--first_fc_hidden_dim', type=int, default=64, help='Embedding dimension for item and user.')
parser.add_argument('--second_fc_hidden_dim', type=int, default=32, help='Embedding dimension for item and user.')

args = parser.parse_args()
'''设置随机种子，使得生成的随机数固定不变'''
torch.manual_seed(args.seed)


'''设置运行的cuda'''
GPU = torch.cuda.is_available()
mdevice = torch.device('cuda:1' if GPU else 'cpu')
args.mdevice = mdevice
args.n_item = 3846  # lastfm-20
args.m_user = 1872
# args.n_item = 3953  # movielens-1m
# args.m_user = 6041
# args.n_item = 8000  # book_crossing
# args.m_user = 2947

item_size = 14  # 7*2

'''training-set和testing-set按照头部尾部用户区分，training-set为头部用户，
testing-set为尾部用户，他们的user_id（一个user_id对应多个item_id）是不同的；
而training-set里面的support-set和query-set是按照一条user_id：[item_id1，...]来划分的，
前20个item_id划分给support-set，后面的12个item_id划分给query-set，所以support-set和query-set
都有同样的user_id。support-set的历史点击记录可以用到query-set中。'''
print("=====================training-set-start=======================")
training_supp_path = args.data_dir + '/training/top_supp_35_pos.txt'
train_dataset = '/training'
training_supp_dataset = LoaderDataset(args, training_supp_path, train_dataset)
print("training_supp_items:" + str(training_supp_dataset.n_item))
print("training_supp_users:" + str(training_supp_dataset.m_user))
training_supp_size = len(training_supp_dataset.trainUniqueItems)   # 训练集有100条数据
print("====training_supp_size====:" + str(training_supp_size))

training_query_path = args.data_dir + '/training/top_query_7_pos.txt'
training_query_dataset = LoaderDataset(args, training_query_path, train_dataset)
print("training_query_items:" + str(training_query_dataset.n_item))
print("training_query_users:" + str(training_query_dataset.m_user))
training_query_size = len(training_query_dataset.trainUniqueItems)
print("====training_query_size====:" + str(training_query_size))
print("=====================training-set-end=======================")

print("=====================testing-set-start=======================")
testing_supp_path = args.data_dir + '/testing/tail_supp_7_pos.txt'
test_dataset = '/testing'
testing_supp_dataset = LoaderDataset(args, testing_supp_path, test_dataset)
print("testing_supp_items:" + str(testing_supp_dataset.n_item))
print("testing_supp_users:" + str(testing_supp_dataset.m_user))
testing_supp_size = len(testing_supp_dataset.trainUniqueItems)   # 测试集有400条数据
print("=====testing_supp_size=====:" + str(testing_supp_size))

testing_query_path = args.data_dir + '/testing/tail_query_7_pos.txt'
testing_query_dataset = LoaderDataset(args, testing_query_path, test_dataset)
print("testing_query_items:" + str(testing_query_dataset.n_item))
print("testing_query_users:" + str(testing_query_dataset.m_user))
testing_query_size = len(testing_query_dataset.trainUniqueItems)   # 测试集有400条数据
print("=====testing_query_size=====:" + str(testing_query_size))
print("=====================testing-set-end=======================")


Recmodel = BaseModel(args, training_supp_dataset, testing_supp_dataset)
Recmodel = Recmodel.to(mdevice)
meta_optim = torch.optim.Adam(Recmodel.parameters(), lr=args.meta_lr, weight_decay=0.0001)
bpr = BPRLossTraining(Recmodel, args)


'''输入头部user_id的list，和尾部需要增强的user_id，输出最高相似分数和与尾部user_id最相似的头部top_user_id，
这里使用的相似度计算法是欧式距离：dist=1/(1+sqrt(pos_user-hist_user)^2)'''
def attention_scores(hist_list, pos_user):
    score_list = []
    for i in range(len(hist_list)):
        dist = torch.sqrt(torch.pow(torch.tensor(pos_user-hist_list[i]), 2))
        sim = 1/(1 + dist)
        score_list.append(sim.item())
    sim_scores = torch.tensor(score_list)
    _scores, recom_indexs = sim_scores.view(-1).sort(descending=True)

    return _scores[0].item(), recom_indexs[0].item()

'''得到增强第test_item_index[j]个item对应的user_list，原来有7个users，现在有14个，增加了1倍数量，
同时需要取得对应的负样本，一共28个user，是经过了gcn的embedding表示'''
def model_test_training(args, training_supp_dataset, testing_supp_dataset, Recmodel, bpr):
    """"训练集train-set进行3层gcn后得到的embedding表示,
    train_items_gcn=torch.Size([3846, 64])，
    train_users_gcn=torch.Size([1872, 64])"""
    train_items_gcn, train_users_gcn = Recmodel.computer_graph_embs("train-set")
    '''测试集test-set进行3层gcn后得到的embedding表示'''
    test_items_gcn, test_users_gcn = Recmodel.computer_graph_embs("test-set")

    train_allPos = training_supp_dataset.allPos  # 训练集training-set所有的正样本item_id
    test_allPos = testing_supp_dataset.allPos  # 测试集testing-set所有的正样本item_id
    item_num = testing_supp_dataset.n_item  # item的数量,这里是3846
    items = random.sample(range(0, testing_supp_dataset.n_item), item_num)  # 随机生成item_num个item_id，范围是（0, dataset.n_item）

    '''获取training-set中support-set的item:{user1,user2,...}'''
    train_all_user_dict = {}
    train_item_index = []
    for i, item in enumerate(items):
        train_pos_user = train_allPos[i]
        if len(train_pos_user) == 0:
            continue
        train_all_user_dict[i] = train_pos_user
        train_item_index.append(i)
    '''获取testing-set中support-set的item:{user1,user2,...}'''
    test_all_user_dict = {}
    test_item_index = []
    for m, item in enumerate(items):
        test_pos_user = test_allPos[m]
        if len(test_pos_user) == 0:
            continue
        test_all_user_dict[m] = test_pos_user
        test_item_index.append(m)

    total_loss_list = []

    for j in range(len(test_item_index)):   # 400次循环
        test_i = test_item_index[j]     # 对第一个item_id进行操作
        test_list = test_all_user_dict[test_i]
        test_list = test_list.tolist()
        test_users_gcn_emb = test_users_gcn[test_list]
        test_users_gcn_emb = test_users_gcn_emb.tolist()

        t_score, t_index = attention_scores(train_item_index, test_i)

        '''因为test_i的数量是train_i的4倍，所以用k = j % 100表示循环4次取train_i的index.
        这里进行了尾部数据增强操作，样本数量从7个增强到14个'''
        # k = j % 100
        # train_i = train_item_index[t_index]
        # train_list = train_all_user_dict[train_i]
        # sim_score_list = []
        # for index in range(len(test_list)):
        #     test_pos_user = test_list[index]
        #     train_list_emb = train_users_gcn[train_list]
        #
        #     sim_score, sim_index = attention_scores(train_list, test_pos_user)
        #     sim_score_list.append(round(sim_score, 4))      # 保留小数点后4位
        #     test_list.append(train_list[sim_index])
        #     test_users_gcn_emb.append((train_list_emb[sim_index]).tolist())

        neg_user_emb_list = []
        neg_user_list = []
        '''抽取负样本，14（正）+14（负）'''
        for t in range(len(test_list)):     # 给14个user补充负样本
            while True:
                neg_user = np.random.randint(0, testing_supp_dataset.m_user)
                if neg_user in test_list:
                    continue
                else:
                    break
            '''对补充后的正样本对应的负样本也要乘以一个sim_score'''
            neg_user_list.append(neg_user)
            neg_user_emb_list.append(test_users_gcn[neg_user].tolist())

        '''一共28个user的gcn的embedding表示，前14个为正样本，后14个为负样本，并没有打乱顺序'''
        pos_user_emb_list = torch.tensor(test_users_gcn_emb)
        neg_user_emb_list = torch.tensor(neg_user_emb_list)

        hist_user_emb_list = []
        test_i_list = []
        for n in range(len(test_users_gcn_emb)):
            test_i_list.append(test_i)
            temp_emb = test_users_gcn_emb.copy()
            temp_emb.remove(temp_emb[n])

            hist_emb = temp_emb
            hist_user_emb_list.append(hist_emb)

        item_emb_list = test_items_gcn[test_i_list]
        item_emb_ego = Recmodel.embedding_item(torch.tensor(test_i_list).to(args.mdevice))
        pos_emb_ego = Recmodel.embedding_user(torch.tensor(test_list).to(args.mdevice))
        neg_emb_ego = Recmodel.embedding_user(torch.tensor(neg_user_list).to(args.mdevice))

        hist_emb_list = torch.tensor(hist_user_emb_list)
        pos_emb_list = pos_user_emb_list
        neg_emb_list = neg_user_emb_list
        user_his_emb = Recmodel.attention_layer(hist_emb_list.to(args.mdevice), pos_emb_list.to(args.mdevice))


        reg_loss = (1 / 2) * (item_emb_ego.norm(2).pow(2) + pos_emb_ego.norm(2).pow(2)
                              + neg_emb_ego.norm(2).pow(2)) / float(len(items))

        '''正样本得分'''
        pos_scores = torch.mul(item_emb_list.to(args.mdevice), pos_emb_list.to(args.mdevice)) + torch.mul(user_his_emb.to(args.mdevice), pos_emb_list.to(args.mdevice))
        pos_scores = torch.sum(pos_scores, dim=1)
        '''负样本得分'''
        neg_scores = torch.mul(item_emb_list.to(args.mdevice), neg_emb_list.to(args.mdevice)) + torch.mul(user_his_emb.to(args.mdevice), neg_emb_list.to(args.mdevice))
        neg_scores = torch.sum(neg_scores, dim=1)

        _loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

        reg_loss = reg_loss * args.decay
        _loss = _loss + reg_loss

        bpr.opt.zero_grad()
        _loss.backward(retain_graph=True)
        bpr.opt.step()
        total_loss_list.append(_loss.item())

    aver_loss = sum(total_loss_list)/len(total_loss_list)

    return aver_loss


max_ndcg10 = 0.
max_prec10 = 0.
max_ndcg8 = 0.
max_prec8 = 0.
max_ndcg5 = 0.
max_prec5 = 0.

for epoch in range(args.epochs):

    test_start = time()
    """meta-learning，寻找出比较合适的θ"""
    '''这步model_training必须有，使用BPR损失函数来寻找最优模型参数'''
    training_supp_loss = model_training(args, training_supp_dataset, Recmodel, bpr, "train-set")
    t_loss = supp_testing(args, training_query_dataset, Recmodel, meta_optim, item_size)
    train_end = time()
    print(f'EPOCH[{epoch + 1}/{args.epochs}] Loss[{t_loss:.3f}] Time[{train_end - test_start:.3f}]')


    """local-learning,寻找最合适的θ1，θ2，...，θn"""
    testing_aver_loss = model_test_training(args, training_supp_dataset, testing_supp_dataset, Recmodel, bpr)
    test_pre10, test_ndcg10, test_pre8, test_ndcg8, test_pre5, test_ndcg5 = query_testing(
        args, testing_query_dataset, Recmodel, item_size)

    test_end = time()

    if test_pre10 > max_prec10:
        max_prec10 = test_pre10
    if test_ndcg10.item() > max_ndcg10:
        max_ndcg10 = test_ndcg10.item()
    if test_pre8 > max_prec8:
        max_prec8 = test_pre8
    if test_ndcg8.item() > max_ndcg8:
        max_ndcg8 = test_ndcg8.item()
    if test_pre5 > max_prec5:
        max_prec5 = test_pre5
    if test_ndcg5.item() > max_ndcg5:
        max_ndcg5 = test_ndcg5.item()
    if epoch % 10 == 0:
        print("=============================query-testing-start=============================")
        print("TOP-10: query_prec:{:.4f}\t\tquery_ndcg:{:.4f}".format(test_pre10, test_ndcg10.item()))
        print("TOP-8: query_prec:{:.4f}\t\tquery_ndcg:{:.4f}".format(test_pre8, test_ndcg8.item()))
        print("TOP-5: query_prec:{:.4f}\t\tquery_ndcg:{:.4f}".format(test_pre5, test_ndcg5.item()))
        print("--------------------------------------------------------------------------------")
        print("TOP-10: max_prec:{:.4f}\t\tmax_ndcg:{:.4f}".format(max_prec10, max_ndcg10))
        print("TOP-8: max_prec:{:.4f}\t\tmax_ndcg:{:.4f}".format(max_prec8, max_ndcg8))
        print("TOP-5: max_prec:{:.4f}\t\tmax_ndcg:{:.4f}".format(max_prec5, max_ndcg5))
        print("===========================query-testing-end=================================")