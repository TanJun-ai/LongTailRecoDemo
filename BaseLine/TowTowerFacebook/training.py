
import torch
from time import time

from BaseLine.TowTowerFacebook.load_dataset import LoadSample
from BaseLine.TowTowerFacebook.metrics import shuffle, minibatch, add_metric


class BPRLoss(torch.nn.Module):
    """bpr算法"""
    def __init__(self):
        super().__init__()

    def forward(self, pos_score, neg_score):
        loss = torch.mean(-(pos_score - neg_score).sigmoid().log(), dim=-1)
        return loss

'''对所有数据进行一次训练'''
def model_training(args, epoch, dataset, recommend_model):

    start = time()
    Recmodel = recommend_model
    Recmodel.train()    # 模型训练
    optimizer = torch.optim.Adam(Recmodel.parameters(), lr=args.lr)
    criterion = BPRLoss()

    '''加载和处理数据,得到S=[[item,user+,user-],...]'''
    S = LoadSample(dataset, "train")

    items_id = torch.Tensor(S[:, 0]).long()
    pos_users = torch.Tensor(S[:, 1]).long()
    neg_users = torch.Tensor(S[:, 2]).long()

    items_id = items_id.to(args.mdevice)
    pos_users = pos_users.to(args.mdevice)
    neg_users = neg_users.to(args.mdevice)

    '''utils.shuffle用于随机打乱顺序，但对应关系不变'''
    items_id, pos_users, neg_users = shuffle(items_id, pos_users, neg_users)

    total_loss = []
    '''将组合矩阵(items_id, pos_users_id, neg_items_id, history_users_id）分成batch_i批,进行分批训练'''
    for (batch_i, (batch_items, batch_pos, batch_neg)) in \
            enumerate(minibatch(items_id, pos_users, neg_users, batch_size=args.batch_size)):

        items_list = batch_items.tolist()
        pos_list = batch_pos.tolist()
        neg_list = batch_neg.tolist()

        item_one_hot, pos_one_hot, neg_one_hot = Recmodel.get_one_hot_emb(items_list, pos_list, neg_list)
        '''计算正负样本的得分'''
        pos_score, neg_score = Recmodel(item_one_hot, pos_one_hot, neg_one_hot)

        loss = criterion(pos_score, neg_score)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss.append(loss)

    aver_loss = sum(total_loss) / len(total_loss)
    end = time()
    print(f"Epoch:{epoch} train_loss:{aver_loss:.4f} spend_times:{(end-start):.2f}")

'''对所有数据进行一次测试'''
def model_testing(args, dataset, recommend_model, item_size):
    Recmodel = recommend_model
    Recmodel.eval()    # 模型测试

    '''加载和处理数据'''
    S = LoadSample(dataset, "test")

    items_id = torch.Tensor(S[:, 0]).long()
    users_id = torch.Tensor(S[:, 1]).long()
    labels = torch.Tensor(S[:, 2]).long()

    items_id = items_id.to(args.mdevice)
    users_id = users_id.to(args.mdevice)
    labels = labels.to(args.mdevice)

    test_data_len = len(dataset.trainUniqueItems)

    total_ndcg_list = []
    total_prec_list = []

    for i in range(test_data_len):
        batch_items = items_id[i*item_size:(i+1)*item_size]
        batch_users = users_id[i*item_size:(i+1)*item_size]
        batch_labels = labels[i*item_size:(i+1)*item_size]

        # print("----------------batch_items------------------")
        # print(batch_items)
        # print("===========batch_users=============")
        # print(batch_users)
        # print("===========batch_labels===========")
        # print(batch_labels)

        batch_items, batch_users, batch_labels = shuffle(batch_items, batch_users, batch_labels)

        items_list = batch_items.tolist()
        users_list = batch_users.tolist()

        item_one_hot, user_pos, user_pos_copy = Recmodel.get_one_hot_emb(items_list, users_list, users_list)

        pred_score, _ = Recmodel(item_one_hot, user_pos, user_pos_copy)

        test_output_list, query_recom_list = pred_score.view(-1).sort(descending=True)
        test_prec, test_ndcg = add_metric(query_recom_list, batch_labels, topn=10)
        total_prec_list.append(test_prec)
        total_ndcg_list.append(test_ndcg)

    total_prec = sum(total_prec_list) / len(total_prec_list)
    total_ndcg = sum(total_ndcg_list) / len(total_ndcg_list)

    return total_prec, total_ndcg