
import argparse
import torch
import numpy as np
import json
from BaseLine.TowTowerFacebook.dssm_facebook import FaceBookDSSM
from BaseLine.TowTowerFacebook.load_dataset import LoaderDataset
from BaseLine.TowTowerFacebook.training import model_training, model_testing

parser = argparse.ArgumentParser()

# parser.add_argument('--data_path', default='../data/lastfm-20-100-400/')
parser.add_argument('--data_path', default='../data/movielens-1m-400-1600/')
# parser.add_argument('--data_path', default='../data/book_crossing-400-1600/')

parser.add_argument('--emb_output_size', default=64, type=int, help='embedding层输出大小')
parser.add_argument('--num_epoch', default=500, type=int, help='跑实验的循环次数')
parser.add_argument('--lr', default=0.001, type=float, help='学习率')
parser.add_argument('--batch_size', default=248, type=int, help='最小批的大小')

args = parser.parse_args()

'''将int型转成one-hot向量'''
def to_onehot_dict(e_list):
    e_dict = {}
    length = len(e_list)
    for index, element in enumerate(e_list):
        vector = torch.zeros(1, length).long()
        element = int(element)
        vector[:, element] = 1.0
        e_dict[element] = vector
    return e_dict

with open('{}/item_list.json'.format(args.data_path), 'r', encoding='utf-8') as f:
    itemids = json.loads(f.read())
with open('{}/user_list.json'.format(args.data_path), 'r', encoding='utf-8') as g:
    userids = json.loads(g.read())
item_dict = to_onehot_dict(itemids)
user_dict = to_onehot_dict(userids)

np.random.seed(2022)
torch.manual_seed(2022)

GPU = torch.cuda.is_available()
mdevice = torch.device('cuda:0' if GPU else 'cpu')
args.mdevice = mdevice

# args.n_item = 3846  # lastfm-20
# args.m_user = 1872
# item_size = 14  # 7*2
args.n_item = 3953    # movielens-1m
args.m_user = 6041
item_size = 20  # 10*2
# args.n_item = 8000    # book_crossing
# args.m_user = 2947
# item_size = 14  # 7*2

train_path = args.data_path+'testing/tail_supp_7_pos.txt'
test_path = args.data_path+'testing/tail_query_10_pos.txt'

print("=========================Load_dataset=========================")
train_dataset = LoaderDataset(args, train_path)
test_dataset = LoaderDataset(args, test_path)

print("=========================Two-Tower-Model=========================")
user_params = {"dims": [256, 128, 64, 32]}
item_params = {"dims": [256, 128, 64, 32]}
temperature = 0.02
two_tower_model = FaceBookDSSM(args, user_params, item_params, item_dict, user_dict, temperature).to(args.mdevice)
print(two_tower_model)
print("-------------------------------------------------------")

max_prec = 0.
max_ndcg = 0.

for epoch in range(args.num_epoch):

    model_training(args, epoch, train_dataset, two_tower_model)

    total_prec, total_ndcg = model_testing(args, test_dataset, two_tower_model, item_size)

    if total_prec > max_prec:
        max_prec = total_prec
    if total_ndcg.item() > max_ndcg:
        max_ndcg = total_ndcg.item()
    if epoch % 10 == 0:
        print("----------------------query-testing---------------------")
        print("TOP-10: query_prec:{:.4f}\t\tquery_ndcg:{:.4f}".format(total_prec, total_ndcg.item()))
        print("max_prec:{:.4f}\t\tmax_ndcg:{:.4f}".format(max_prec, max_ndcg))
        print("-------------------------------------------------------")