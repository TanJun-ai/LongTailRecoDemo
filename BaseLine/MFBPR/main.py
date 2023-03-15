
import argparse
import torch
import numpy as np
from BaseLine.MFBPR.training import model_training, model_testing
from BaseLine.MFBPR.load_dataset import LoaderDataset
from BaseLine.MFBPR.BPRMF import BPRMF

parser = argparse.ArgumentParser()

# parser.add_argument('--data_path', default='../data/lastfm-20-100-400/')
# parser.add_argument('--data_path', default='../data/movielens-1m-400-1600/')
parser.add_argument('--data_path', default='../data/book_crossing-400-1600/')

parser.add_argument('--emb_size', default=64, type=int, help='embedding层输出大小')
parser.add_argument('--num_epoch', default=500, type=int, help='跑实验的循环次数')
parser.add_argument('--lr', default=0.001, type=float, help='学习步长')
parser.add_argument('--decay', default=0.001, type=float, help='梯度')
parser.add_argument('--weigh_decay', default=0.001, type=float, help='梯度')
parser.add_argument('--layers', default=3, type=int, help='卷积层的数量')
parser.add_argument('--batch_size', default=1024, type=int, help='最小批的大小')
parser.add_argument('--top_k', default=5, type=int,
                    help='这个参数是选择top k中的k的，即上面是用来查看前10个商品，和前20个商品的分数排名')

args = parser.parse_args()

np.random.seed(2022)
torch.manual_seed(2022)

GPU = torch.cuda.is_available()
mdevice = torch.device('cuda:1' if GPU else 'cpu')
args.mdevice = mdevice

# args.n_item = 3846    # lastfm-20
# args.m_user = 1872
# item_size = 14  # 7*2
# args.n_item = 3953    # movielens-1m
# args.m_user = 6041
# item_size = 20  # 10*2
args.n_item = 8000    # book_crossing
args.m_user = 2947
item_size = 14  # 7*2


train_path = args.data_path+'testing/tail_supp_7_pos.txt'
test_path = args.data_path+'testing/tail_query_7_pos.txt'

print("=========================Load_dataset=========================")
train_dataset = LoaderDataset(args, train_path)
test_dataset = LoaderDataset(args, test_path)

print("=========================BPRMF=========================")
bpr_model = BPRMF(args.n_item, args.m_user, args).to(args.mdevice)
print(bpr_model)
print("-------------------------------------------------------")

max_prec = 0.
max_ndcg = 0.

for epoch in range(args.num_epoch):

	model_training(args, epoch, train_dataset, bpr_model)

	total_prec, total_ndcg = model_testing(args, test_dataset, bpr_model, item_size)

	if total_prec > max_prec:
		max_prec = total_prec
	if total_ndcg.item() > max_ndcg:
		max_ndcg = total_ndcg.item()
	if epoch % 10 == 0:
		print("----------------------query-testing---------------------")
		print("TOP-"+str(args.top_k)+": query_prec:{:.4f}\t\tquery_ndcg:{:.4f}".format(total_prec, total_ndcg.item()))
		print("max_prec:{:.4f}\t\tmax_ndcg:{:.4f}".format(max_prec, max_ndcg))
		print("-------------------------------------------------------")
