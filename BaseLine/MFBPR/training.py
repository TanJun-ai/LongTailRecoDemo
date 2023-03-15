
import torch
from time import time

from BaseLine.MFBPR.load_dataset import LoadSample
from BaseLine.MFBPR.utils import shuffle, minibatch, add_metric

'''对所有数据进行一次训练'''
def model_training(args, epoch, dataset, recommend_model):

    start = time()
    Recmodel = recommend_model
    Recmodel.train()    # 模型训练
    optimizer = torch.optim.Adam(Recmodel.parameters(), lr=args.lr)

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

    num_batch = len(items_id) // args.batch_size + 1  # args.bpr_batch=2048
    aver_loss = 0.
    '''将组合矩阵(items_id, pos_users_id, neg_items_id, history_users_id）分成batch_i批,进行分批训练'''
    for (batch_i, (batch_items, batch_pos, batch_neg)) in \
            enumerate(minibatch(items_id, pos_users, neg_users, batch_size=args.batch_size)):

        u_emb, pos_emb, neg_emb, u_emb_ego, pos_emb_ego, neg_emb_ego = \
            Recmodel.get_embedding(batch_items, batch_pos, batch_neg)
        reg_loss = (1 / 2) * (u_emb_ego.norm(2).pow(2) + pos_emb_ego.norm(2).pow(2) +
                              neg_emb_ego.norm(2).pow(2)) / float(len(batch_items))
        reg_loss = reg_loss * args.weigh_decay

        pos_scores = Recmodel(u_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)

        neg_scores = Recmodel(u_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

        loss = loss + reg_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        aver_loss += loss

    aver_loss = aver_loss / num_batch
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

    """这里的测试是按照一个item_id对应其所有交互的user_id来进行的，例如：
    37: [1240 483 149 2 658 246 95 539 628 1689 1800 53 47 179]
    会先预测item_id=37的user_id列表，然后对比其label，计算precision和ndcg"""
    for i in range(test_data_len):
        batch_items = items_id[i*item_size:(i+1)*item_size]
        batch_users = users_id[i*item_size:(i+1)*item_size]
        batch_labels = labels[i*item_size:(i+1)*item_size]

        batch_items, batch_users, batch_labels = shuffle(batch_items, batch_users, batch_labels)
        u_emb, user_emb, u_emb, u_emb_ego, u_emb_ego, u_emb_ego = \
            Recmodel.get_embedding(batch_items, batch_users, batch_users)

        pred_score = Recmodel(u_emb, user_emb)
        pred_score = torch.sum(pred_score, dim=1)

        test_output_list, query_recom_list = pred_score.view(-1).sort(descending=True)
        test_prec, test_ndcg = add_metric(query_recom_list, batch_labels, topn=args.top_k)
        total_prec_list.append(test_prec)
        total_ndcg_list.append(test_ndcg)


    total_prec = sum(total_prec_list) / len(total_prec_list)
    total_ndcg = sum(total_ndcg_list) / len(total_ndcg_list)

    return total_prec, total_ndcg