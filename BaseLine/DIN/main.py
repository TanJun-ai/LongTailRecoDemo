import numpy as np
import argparse
import torch

from BaseLine.DIN.training import model_training, model_testing
from din_model import DIN
from load_dataset import LoaderDataset

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='../data/lastfm-20-100-400')
# parser.add_argument('--data_dir', type=str, default='../data/movielens-1m-100-400')
# parser.add_argument('--data_dir', type=str, default='../data/book_crossing-400-1600')

parser.add_argument('--item_dim', type=int, default=3846, help='Embedding dimension for item')    # lastfm-20
parser.add_argument('--user_dim', type=int, default=1872, help='Embedding dimension for item')
# parser.add_argument('--item_dim', type=int, default=3953, help='Embedding dimension for item')    # movielens-1m
# parser.add_argument('--user_dim', type=int, default=6041, help='Embedding dimension for item')

parser.add_argument('--emb_out_dim', type=int, default=32, help='Embedding output dimension')
parser.add_argument('--input_dim', type=int, default=96, help='3 emb,32x3=96')
parser.add_argument('--first_embedding_dim', type=int, default=64, help='Embedding dimension for MLP.')
parser.add_argument('--second_embedding_dim', type=int, default=32, help='Embedding dimension for MLP')

parser.add_argument('--dropout', type=float, default=0., help='dropout rate for MLP')
parser.add_argument('--lr', type=float, default=1e-4, help='Applies to SGD and Adagrad.')
parser.add_argument('--optim', type=str, default='adam', help='sgd, adagrad, adam or adamax.')
parser.add_argument('--num_epoch', type=int, default=500)
parser.add_argument('--batch_size', type=int, default=64)

args = parser.parse_args()


np.random.seed(2022)
torch.manual_seed(2022)

GPU = torch.cuda.is_available()
mdevice = torch.device('cuda:0' if GPU else 'cpu')
args.mdevice = mdevice

args.n_item = 3846    # lastfm-20
args.m_user = 1872
# args.n_item = 3953    # movielens-1m
# args.m_user = 6041
# args.n_item = 8000    # book_crossing
# args.m_user = 2947

print("=====================training-set-start=======================")
supp_path = args.data_dir + '/testing/tail_supp_7_pos.txt'
supp_dataset = LoaderDataset(args, supp_path)
print("=====================training-set-end=======================")

print("=====================testing-set-start=======================")
# query_path = args.data_dir + '/testing/tail_query_15_pos.txt'
query_path = args.data_dir + '/testing/tail_query_7_pos.txt'
query_dataset = LoaderDataset(args, query_path)
# item_size = 30  # 15*2
item_size = 14  # 7*2
print("=====================testing-set-end=======================")

print("======================Recmodel=============================")
Recmodel = DIN(args)
Recmodel.to(args.mdevice)
print(Recmodel)
print("----------------------------------------------------------")

max_prec = 0.
max_ndcg = 0.

for epoch in range(args.num_epoch):
    train_output_info = model_training(args, supp_dataset, Recmodel)
    print("Epoch:"+str(epoch)+'\t\t'+train_output_info)

    total_prec, total_ndcg = model_testing(args, query_dataset, Recmodel, item_size)

    if total_prec > max_prec:
        max_prec = total_prec
    if total_ndcg.item() > max_ndcg:
        max_ndcg = total_ndcg.item()
    if epoch % 10 == 0:
        print("----------------------query-testing---------------------")
        print("TOP-10: query_prec:{:.4f}\t\tquery_ndcg:{:.4f}".format(total_prec, total_ndcg.item()))
        print("max_prec:{:.4f}\t\tmax_ndcg:{:.4f}".format(max_prec, max_ndcg))
        print("-------------------------------------------------------")

