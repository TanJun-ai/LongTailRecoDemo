
import os
from time import time
import numpy as np
import random
import argparse
import torch
import json
from TaNP import Trainer
from model_training import training


parser = argparse.ArgumentParser()
# parser.add_argument('--data_dir', type=str, default='../data/lastfm-20-100-400')
# parser.add_argument('--data_dir', type=str, default='../data/movielens-1m-200-800-random')
# parser.add_argument('--data_dir', type=str, default='../data/book_crossing-400-1600')
parser.add_argument('--data_dir', type=str, default='../data/book_crossing_100_400')

parser.add_argument('--first_embedding_dim', type=int, default=32, help='Embedding dimension for item and user.')
parser.add_argument('--second_embedding_dim', type=int, default=16, help='Embedding dimension for item and user.')
parser.add_argument('--z1_dim', type=int, default=32, help='The dimension of z1 in latent path.')
parser.add_argument('--z2_dim', type=int, default=32, help='The dimension of z2 in latent path.')
parser.add_argument('--z_dim', type=int, default=32, help='The dimension of z in latent path.')

parser.add_argument('--enc_h1_dim', type=int, default=64, help='The hidden first dimension of encoder.')
parser.add_argument('--enc_h2_dim', type=int, default=64, help='The hidden second dimension of encoder.')

parser.add_argument('--taskenc_h1_dim', type=int, default=128, help='The hidden first dimension of task encoder.')
parser.add_argument('--taskenc_h2_dim', type=int, default=64, help='The hidden second dimension of task encoder.')
parser.add_argument('--taskenc_final_dim', type=int, default=64, help='The hidden second dimension of task encoder.')
parser.add_argument('--clusters_k', type=int, default=7, help='Cluster numbers of tasks.')
parser.add_argument('--temperature', type=float, default=1.0, help='used for student-t distribution.')
parser.add_argument('--lambda', type=float, default=0.1, help='used to balance the clustering loss and NP loss.')

parser.add_argument('--dec_h1_dim', type=int, default=128, help='The hidden first dimension of encoder.')
parser.add_argument('--dec_h2_dim', type=int, default=128, help='The hidden second dimension of encoder.')
parser.add_argument('--dec_h3_dim', type=int, default=128, help='The hidden third dimension of encoder.')
parser.add_argument('--dropout_rate', type=float, default=0, help='used in encoder and decoder.')
parser.add_argument('--lr', type=float, default=1e-4, help='Applies to SGD and Adagrad.')
parser.add_argument('--optim', type=str, default='adam', help='sgd, adagrad, adam or adamax.')
parser.add_argument('--num_epoch', type=int, default=500)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--train_ratio', type=float, default=0.7, help='Warm user ratio for training_min.')
parser.add_argument('--valid_ratio', type=float, default=0.1, help='Cold user ratio for validation.')
parser.add_argument('--seed', type=int, default=2022)
parser.add_argument('--save', type=int, default=0)
parser.add_argument('--use_cuda', type=bool, default=torch.cuda.is_available())
parser.add_argument('--cpu', action='store_true', help='Ignore CUDA.')
parser.add_argument('--support_size', type=int, default=20)
parser.add_argument('--query_size', type=int, default=10)
parser.add_argument('--max_len', type=int, default=200, help='The max length of interactions for each user.')
parser.add_argument('--context_min', type=int, default=20, help='Minimum size of context range.')
args = parser.parse_args()

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

def seed_everything(seed=2022):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed = args.seed
seed_everything(seed)

GPU = torch.cuda.is_available()
mdevice = torch.device('cuda:0' if GPU else 'cpu')
args.mdevice = mdevice
opt = vars(args)


# opt['uf_dim'] = 3846    # lastfm-20
# opt['if_dim'] = 1872
# opt['uf_dim'] = 3953  # movielens-1m
# opt['if_dim'] = 6041
opt['uf_dim'] = 8000  # book_crossing
opt['if_dim'] = 2947

trainer = Trainer(opt)
trainer.to(args.mdevice)


start = time()
print("=======================training-set-start=========================")
with open('{}/item_list.json'.format(opt["data_dir"]), 'r', encoding='utf-8') as f:
    itemids = json.loads(f.read())
with open('{}/user_list.json'.format(opt["data_dir"]), 'r', encoding='utf-8') as g:
    userids = json.loads(g.read())
item_dict = to_onehot_dict(itemids)
user_dict = to_onehot_dict(userids)

top_supp_path = opt["data_dir"] + '/training/top_supp_70.txt'
top_supp_label_path = opt["data_dir"] + '/training/top_supp_70_label.txt'
top_query_path = opt["data_dir"] + '/training/top_query_14.txt'
top_query_label_path = opt["data_dir"] + '/training/top_query_14_label.txt'

training_set_size = 100

'''movielens-1m：supp_xs_s包含了675条记录，每条记录里面100个item_id+user_id，
[0, 0, 0,  ..., 0, 0, 0]前面的6040个为item_id，后面的3952个为user_id，每个item_id、user_id都用one-hot向量表示'''
supp_xs_s = []
supp_ys_s = []
query_xs_s = []
query_ys_s = []

for idx in range(training_set_size):
    supp_x_cache = []  # 暂时缓存的每一列U-I总数据
    support_x_total = None  # 将100条user_id+item_id串联起来
    with open(top_supp_path) as fo:
        for lt in fo.readlines():
            if len(lt) > 0:
                lt = lt.strip().strip('\n').split(' ')
                supp_x_cache.append(lt)
    for j in range(len(supp_x_cache[idx]) - 1):
        item_id = int(supp_x_cache[idx][0])
        user_id = int(supp_x_cache[idx][j + 1])
        temp_x_tensoe = torch.cat((user_dict[user_id], item_dict[item_id]), 1)
        # temp_x_tensoe = torch.cat((item_dict[item_id], user_dict[user_id]), 1)
        try:
            support_x_total = torch.cat((support_x_total, temp_x_tensoe), 0)
        except:
            support_x_total = temp_x_tensoe
    supp_xs_s.append(support_x_total)

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
    for n in range(len(query_x_cache[idx]) - 1):
        item_id_q = int(query_x_cache[idx][0])
        user_id_q = int(query_x_cache[idx][n + 1])
        query_x_tensor = torch.cat((user_dict[user_id_q], item_dict[item_id_q]), 1)
        # query_x_tensor = torch.cat((item_dict[item_id_q], user_dict[user_id_q]), 1)
        try:
            query_x_total = torch.cat((query_x_total, query_x_tensor), 0)
        except:
            query_x_total = query_x_tensor
    query_xs_s.append(query_x_total)

    query_y_cache = []
    with open(top_query_label_path) as qe:
        for le in qe.readlines():
            if len(le) > 0:
                le = le.strip().strip('\n').split(' ')
                query_y_cache.append(le)
    query_y_app = torch.FloatTensor([float(x) for x in query_y_cache[idx][1:]])
    query_ys_s.append(query_y_app)

train_dataset = list(zip(supp_xs_s, supp_ys_s, query_xs_s, query_ys_s))
del (supp_xs_s, supp_ys_s, query_xs_s, query_ys_s)
print("item-nums:" + str(opt['uf_dim']))
print("user-nums:" + str(opt['if_dim']))
print("training_set_size:" + str(training_set_size))
print("=========================training-set-end==========================")

print("===============testing-set-start=================")
tail_supp_path = opt["data_dir"] + '/testing/tail_supp_14.txt'
tail_supp_label_path = opt["data_dir"] + '/testing/tail_supp_14_label.txt'
tail_query_path = opt["data_dir"] + '/testing/tail_query_14.txt'
tail_query_label_path = opt["data_dir"] + '/testing/tail_query_14_label.txt'

testing_set_size = 400

supp_xs_s = []
supp_ys_s = []
query_xs_s = []
query_ys_s = []
for idxx in range(testing_set_size):
    supp_x_cache = []  # 暂时缓存的每一列U-I总数据
    support_x_total = None  # 将100条user_id+item_id串联起来
    with open(tail_supp_path) as fo:
        for lt in fo.readlines():
            if len(lt) > 0:
                lt = lt.strip().strip('\n').split(' ')
                supp_x_cache.append(lt)
    for j in range(len(supp_x_cache[idxx]) - 1):
        item_id = int(supp_x_cache[idxx][0])
        user_id = int(supp_x_cache[idxx][j + 1])
        temp_x_tensoe = torch.cat((user_dict[user_id], item_dict[item_id]), 1)
        # temp_x_tensoe = torch.cat((item_dict[item_id], user_dict[user_id]), 1)
        try:
            support_x_total = torch.cat((support_x_total, temp_x_tensoe), 0)
        except:
            support_x_total = temp_x_tensoe
    supp_xs_s.append(support_x_total)

    supp_y_cache = []
    with open(tail_supp_label_path) as su:
        for lu in su.readlines():
            if len(lu) > 0:
                lu = lu.strip().strip('\n').split(' ')
                supp_y_cache.append(lu)
    support_y_app = torch.FloatTensor([float(x) for x in supp_y_cache[idxx][1:]])
    supp_ys_s.append(support_y_app)

    query_x_cache = []
    query_x_total = None
    with open(tail_query_path) as qu:
        for lq in qu.readlines():
            if len(lq) > 0:
                lq = lq.strip().strip('\n').split(' ')
                query_x_cache.append(lq)
    for n in range(len(query_x_cache[idxx]) - 1):
        item_id_q = int(query_x_cache[idxx][0])
        user_id_q = int(query_x_cache[idxx][n + 1])
        query_x_tensor = torch.cat((user_dict[user_id_q], item_dict[item_id_q]), 1)
        # query_x_tensor = torch.cat((item_dict[item_id_q], user_dict[user_id_q]), 1)
        try:
            query_x_total = torch.cat((query_x_total, query_x_tensor), 0)
        except:
            query_x_total = query_x_tensor
    query_xs_s.append(query_x_total)

    query_y_cache = []
    with open(tail_query_label_path) as qe:
        for le in qe.readlines():
            if len(le) > 0:
                le = le.strip().strip('\n').split(' ')
                query_y_cache.append(le)
    query_y_app = torch.FloatTensor([float(x) for x in query_y_cache[idxx][1:]])
    query_ys_s.append(query_y_app)

test_dataset = list(zip(supp_xs_s, supp_ys_s, query_xs_s, query_ys_s))
del (supp_xs_s, supp_ys_s, query_xs_s, query_ys_s)
print("testing_set_size:" + str(testing_set_size))
print("===============testing-set-end=================")
end = time()
print(f'Time[{end - start:.3f}]')
print("# epoch\ttrain_loss\tprecision5\tNDCG5\tMAP5\tprecision7\tNDCG7\tMAP7\tprecision10\tNDCG10\tMAP10")
print("===================================================================================================")


print("Start training......")
training(trainer, train_dataset, test_dataset, batch_size=opt['batch_size'], num_epoch=opt['num_epoch'])

