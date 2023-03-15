
import os
import numpy as np
import random
import argparse
import pickle
import torch
import json
from MeLU import MeLU
from training import training_melu
from time import time
from torch.autograd import Variable

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='../data/movielens-1m-100-400')
# parser.add_argument('--data_dir', type=str, default='../data/lastfm-20-100-400')
# parser.add_argument('--data_dir', type=str, default='../data/book_crossing-400-1600')

parser.add_argument('--model_save_dir', type=str, default='save_model_dir')
parser.add_argument('--id', type=str, default='1', help='used for save hyper-parameters.')
parser.add_argument('--seed', type=int, default=2022)
parser.add_argument('--num_epoch', type=int, default=500)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--save', type=int, default=0)
parser.add_argument('--embedding_dim', type=int, default=32)
parser.add_argument('--first_fc_hidden_dim', type=int, default=64, help='Embedding dimension for item and user.')
parser.add_argument('--second_fc_hidden_dim', type=int, default=32, help='Embedding dimension for item and user.')
parser.add_argument('--context_min', type=int, default=20, help='Minimum size of context range.')
parser.add_argument('--local_lr', type=float, default=5e-6, help='Applies to SGD and Adagrad.')
parser.add_argument('--global_lr', type=float, default=5e-5, help='Applies to global update.')

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

GPU = torch.cuda.is_available()
mdevice = torch.device('cuda:0' if GPU else 'cpu')
args.mdevice = mdevice
print("Preprocess is done.")
print("Create model OriginMeLU...")

with open(args.data_dir+"/item_list.json", 'r') as f:
    item_list = json.loads(f.read())
with open(args.data_dir+"/user_list.json", 'r') as g:
    user_list = json.loads(g.read())
user_lens = len(user_list)  # users的数量，这里是1872
item_lens = len(item_list)  # items的数量，这里为3846

# args.item = item_lens
# args.user = user_lens
args.item = user_lens
args.user = item_lens


'''初始化模型'''
model = MeLU(args, mdevice)
model.to(mdevice)
# item_dict = to_onehot_dict(item_list)
# user_dict = to_onehot_dict(user_list)
item_dict = to_onehot_dict(user_list)
user_dict = to_onehot_dict(item_list)

start = time()
print("===============training-set-start=================")
top_supp_path = args.data_dir + '/training/top_supp_70.txt'
top_supp_label_path = args.data_dir + '/training/top_supp_70_label.txt'
top_query_path = args.data_dir + '/training/top_query_20.txt'
top_query_label_path = args.data_dir + '/training/top_query_20_label.txt'

args.top_k = 10
training_set_size = 100

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
        user_id = int(supp_x_cache[idx][0])
        item_id = int(supp_x_cache[idx][j + 1])
        # temp_x_tensoe = torch.cat((user_dict[user_id], item_dict[item_id]), 1)
        temp_x_tensoe = torch.cat((item_dict[item_id], user_dict[user_id]), 1)
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
        user_id_q = int(query_x_cache[idx][0])
        item_id_q = int(query_x_cache[idx][n + 1])
        # query_x_tensor = torch.cat((user_dict[user_id_q], item_dict[item_id_q]), 1)
        query_x_tensor = torch.cat((item_dict[item_id_q], user_dict[user_id_q]), 1)
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
print("item nums:" + str(args.item))
print("user nums:" + str(args.user))
print("training_set_size:" + str(training_set_size))
print("===============training_min-set-end=================")

print("===============testing-set-start=================")
tail_supp_path = args.data_dir + '/testing/tail_supp_14.txt'
tail_supp_label_path = args.data_dir + '/testing/tail_supp_14_label.txt'
tail_query_path = args.data_dir + '/testing/tail_query_14.txt'
tail_query_label_path = args.data_dir + '/testing/tail_query_14_label.txt'

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
        user_id = int(supp_x_cache[idxx][0])
        item_id = int(supp_x_cache[idxx][j + 1])
        # temp_x_tensoe = torch.cat((user_dict[user_id], item_dict[item_id]), 1)
        temp_x_tensoe = torch.cat((item_dict[item_id], user_dict[user_id]), 1)
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
        user_id_q = int(query_x_cache[idxx][0])
        item_id_q = int(query_x_cache[idxx][n + 1])
        # query_x_tensor = torch.cat((user_dict[user_id_q], item_dict[item_id_q]), 1)
        query_x_tensor = torch.cat((item_dict[item_id_q], user_dict[user_id_q]), 1)
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

print("========Model:OriginMeLU=========")
print(model)
print("=================================")

print("Start training......")
'''开始训练和预测'''
with torch.autograd.set_detect_anomaly(True):
    training_melu(model, train_dataset, test_dataset,
             batch_size=args.batch_size, num_epoch=args.num_epoch, model_save=args.save)


