'''
Top-K Diversity Regularizer

This software may be used only for research evaluation purposes.
For other purposes (e.g., commercial), please contact the authors.

'''

from time import time
import math
import click
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from model import BPR
from utils import *

# Slice the given list into chunks of size n.
def list_chunk(lst, n):
    return [lst[i:i+n] for i in range(0, len(lst), n)]

@click.command()
# @click.option('--data', type=str, default='lastfm-20-100-400', help='Select Dataset')
# @click.option('--data', type=str, default='movielens-1m-200-800', help='Select Dataset')
@click.option('--data', type=str, default='book_crossing-400-1600', help='Select Dataset')
@click.option('--seed', type=int, default=0, help='Set random seed')
@click.option('--reg', type=bool, default=True, help='Use TDR if True')
@click.option('--unmask', type=bool, default=False, help='Use unmask scheme if True')
@click.option('--ut', type=int, default=0, help='Number of unmasking top items')
@click.option('--ur', type=int, default=0, help='Number of unmasking random items')
@click.option('--ep', type=int, default=200, help='Number of total epoch')
@click.option('--reclen', type=int, default=30, help='Number of epoch with reccommendation loss')
@click.option('--dim', type=int, default=32, help='Number of latent factors')
@click.option('--cpu', type=bool, default=False, help='Use CPU while TDR')
@click.option('--dut', type=float, default=0, help='Change on the number of unmasking top items per epoch')
@click.option('--dur', type=float, default=0, help='Change on the number of unmasking random items per epoch')
@click.option('--rbs', type=int, default=0, help='Number of rows in mini batch')
@click.option('--cbs', type=int, default=0, help='Number of columns in mini batch')

def main(data, seed, reg, unmask, ut, ur, ep, reclen, dim, cpu, dut, dur, rbs, cbs):
    set_seed(seed)
    device = DEVICE
    # set hyperparameters
    config = {
        'lr': 1e-3,
        'decay': 1e-4,
        'latent_dim': dim,
        'batch_size': 1028,
        'epochs': ep,
        'ks': [5, 8, 10],
        'train_neg': 4,
        'test_neg': 99,
        # 'n_item': 3846,  # last-fm
        # 'm_user': 1872
        # 'n_item': 3953,  # movielens-1m
        # 'm_user': 6041
        'n_item': 8000,  # book_crossing
        'm_user': 2947
    }
    print("=====================config=========================")
    print(config)
    torch.multiprocessing.set_sharing_strategy('file_system')

    print("====================training-set-start======================")
    train_path = f'../data/{data}/testing/tail_supp_7_pos.txt'
    train_set = LoaderDataset(config, train_path)
    item_num = config['n_item']
    user_num = config['m_user']
    train_mat = train_set.UserItemNet
    print('user_nums:', user_num, 'item_nums:', item_num, 'train_data_len:', len(train_set.trainUniqueItems))
    print("====================training-set-end======================")

    print("====================testing-set-start======================")
    test_path = f'../data/{data}/testing/tail_query_14.txt'
    test_label_path = f'../data/{data}/testing/tail_query_14_label.txt'
    test_set = LoaderDataset(config, test_path)
    testing_query_size = len(test_set.trainUniqueItems)  # 测试集有400条数据
    print("======testing_query_size=======")
    print(testing_query_size)

    test_items_id_list = []
    test_users_id_list = []
    test_labels_list = []
    for i in range(testing_query_size):
        cache_list = []
        with open(test_path) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    cache_list.append(l)
        items_id = []
        for j in range(len(cache_list[i]) - 1):
            items_id.append(int(cache_list[i][0]))
        users_id = []
        users_id.extend(int(i) for i in cache_list[i][1:])
        test_items_id_list.append(torch.Tensor(items_id).long())
        test_users_id_list.append(torch.Tensor(users_id).long())

    for j in range(testing_query_size):
        cache_list = []
        with open(test_label_path) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    cache_list.append(l)
        labels_id = []
        labels_id.extend(float(i) for i in cache_list[j][1:])
        test_labels_list.append(torch.Tensor(labels_id).long())

    print('user_nums:', user_num, 'item_nums:', item_num, 'test_data_len:', len(test_set.trainUniqueItems))
    print("=====================testing-set-end=======================")

    model = BPR(item_num, user_num, config['latent_dim'])
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['decay'])


    max_ndcg = 0.
    max_prec = 0.
    '''开始训练和测试'''
    for epoch in range(1, config['epochs'] + 1):
        start = time()
        model.train()

        S = UniformSample(train_set, 4)
        items_id = torch.Tensor(S[:, 0]).long()
        pos_users_id = torch.Tensor(S[:, 1]).long()
        neg_users_id = torch.Tensor(S[:, 2]).long()
        items_id = items_id.to(DEVICE)
        pos_users_id = pos_users_id.to(DEVICE)
        neg_users_id = neg_users_id.to(DEVICE)

        items_id, pos_users_id, neg_users_id = shuffle(items_id, pos_users_id, neg_users_id)
        '''分批训练'''
        train_loss = []
        for (batch_i, (batch_items, batch_pos, batch_neg)) in \
                enumerate(minibatch(items_id, pos_users_id, neg_users_id, batch_size=config['batch_size'])):
            model.zero_grad()
            prediction_i, prediction_j = model(batch_items, batch_pos, batch_neg)
            rec_loss = - (prediction_i - prediction_j).sigmoid().log().sum()
            rec_loss.backward()
            optimizer.step()
            train_loss.append(rec_loss)

        t_loss = torch.stack(train_loss).mean(0)

        # top-k inference ，'ks': [5, 8, 10]
        k = config['ks'][1]
        '''train with diversity regularizer ,30个epoch之后进行diversity regularizer操作'''
        if reg and epoch > reclen:

            if rbs == 0:
                # row_batch_size = user_num
                row_batch_size = item_num
            else:
                row_batch_size = rbs
            # row_batch = list_chunk(torch.randperm(user_num).tolist(), row_batch_size)
            row_batch = list_chunk(torch.randperm(item_num).tolist(), row_batch_size)
            if cbs == 0:
                # col_batch_size = item_num
                col_batch_size = user_num
            else:
                col_batch_size = cbs
            # col_batch = list_chunk(torch.randperm(item_num).tolist(), col_batch_size)
            col_batch = list_chunk(torch.randperm(user_num).tolist(), col_batch_size)

            # calculate number of unmasking items for each mini batch
            bk = math.ceil(k / len(col_batch))
            bur = math.ceil(max(ur + int((epoch - reclen - 1) * dur), 0) / len(col_batch))
            but = math.ceil(max(ut + int((epoch - reclen - 1) * dut), 0) / len(col_batch))

            for rb in row_batch:
                for cb in col_batch:
                    # inference top-k recommendation lists
                    model.zero_grad()
                    scores = []
                    items = torch.LongTensor(cb).to(device)
                    for u in rb:
                        u = torch.tensor([u]).to(device)
                        score, _ = model(u, items, items)
                        scores.append(score)
                    scores = torch.stack(scores)
                    scores = torch.softmax(scores, dim=1)

                    # unmasking mechanism
                    if unmask:
                        k_ = len(cb) - (bk + but)
                    else:
                        k_ = len(cb) - bk
                    mask_idx = torch.topk(-scores, k=k_)[1]  # mask index for being filled 0
                    if unmask:
                        for u in range(len(rb)):
                            idx = torch.randperm(mask_idx.shape[1])
                            mask_idx[u] = mask_idx[u][idx]
                        if bur > 0:
                            mask_idx = mask_idx[:, :-bur]

                    mask = torch.zeros(size=scores.shape, dtype=torch.bool)
                    mask[torch.arange(mask.size(0)).unsqueeze(1), mask_idx] = True
                    topk_scores = scores.masked_fill(mask.to(device), 0)

                    # coverage regularizer
                    scores_sum = torch.sum(topk_scores, dim=0, keepdim=False)
                    epsilon = 0.01
                    scores_sum += epsilon
                    d_loss = -torch.sum(torch.log(scores_sum))

                    # skewness regularizer
                    topk_scores = torch.topk(scores, k=k)[0]
                    norm_scores = topk_scores / torch.sum(topk_scores, dim=1, keepdim=True)
                    e_loss = torch.sum(torch.sum(norm_scores * torch.log(norm_scores), dim=1))

                    # sum of losses
                    regularizations = d_loss + e_loss
                    regularizations.backward()
                    optimizer.step()

        '''evaluate metrics，test-set'''
        model.eval()
        total_ndcg_list = []
        total_prec_list = []
        for j in range(testing_query_size):
            test_items_id_list[j] = test_items_id_list[j].to(DEVICE)
            test_users_id_list[j] = test_users_id_list[j].to(DEVICE)
            test_labels_list[j] = test_labels_list[j].to(DEVICE)
            y_pred_scores, _ = model(test_items_id_list[j], test_users_id_list[j], test_users_id_list[j])
            output_list, y_recom_list = y_pred_scores.sort(descending=True)

            test_prec, test_ndcg = add_metric(y_recom_list, test_labels_list[j], topn=k)
            total_prec_list.append(test_prec)
            total_ndcg_list.append(test_ndcg)

        total_prec = sum(total_prec_list) / len(total_prec_list)
        total_ndcg = sum(total_ndcg_list) / len(total_ndcg_list)
        if total_prec > max_prec:
            max_prec = total_prec
        if total_ndcg.item() > max_ndcg:
            max_ndcg = total_ndcg.item()
        if epoch % 10 == 0:
            print("----------------------query-testing---------------------")
            print("TOP-"+str(k)+": query_prec:{:.4f}\t\tquery_ndcg:{:.4f}".format(total_prec, total_ndcg.item()))
            print("max_prec:{:.4f}\t\tmax_ndcg:{:.4f}".format(max_prec, max_ndcg))
            print("-------------------------------------------------------")

        end = time()
        print(f'EPOCH[{epoch}] Loss[{t_loss:.3f}] Time[{end - start:.3f}]')

if __name__ == '__main__':
    main()

