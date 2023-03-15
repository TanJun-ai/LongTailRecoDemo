

import torch
from BaseLine.MetaKG.utility.utils import LoadSample
from BaseLine.MetaKG.utility.utils import shuffle, add_metric


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

    entity_gcn_emb, user_gcn_emb = recommend_model.generate()

    """这里的测试是按照一个item_id对应其所有交互的user_id来进行的，例如：
    37: [1240 483 149 2 658 246 95 539 628 1689 1800 53 47 179]
    会先预测item_id=37的user_id列表，然后对比其label，计算precision和ndcg"""
    for i in range(test_data_len):
        batch_items = items_id[i*item_size:(i+1)*item_size]
        batch_users = users_id[i*item_size:(i+1)*item_size]
        batch_labels = labels[i*item_size:(i+1)*item_size]

        batch_items, batch_users, batch_labels = shuffle(batch_items, batch_users, batch_labels)

        u_g_embeddings = user_gcn_emb[batch_items]
        i_g_embddings = entity_gcn_emb[batch_users]

        i_batch_ratings = recommend_model.rating(u_g_embeddings, i_g_embddings).to(args.mdevice)

        pred_score = (sum(i_batch_ratings) / len(i_batch_ratings)).view(-1, 1)

        test_output_list, query_recom_list = pred_score.view(-1).sort(descending=True)

        test_prec, test_ndcg = add_metric(query_recom_list, batch_labels, topn=10)
        total_prec_list.append(test_prec)
        total_ndcg_list.append(test_ndcg)


    total_prec = sum(total_prec_list) / len(total_prec_list)
    total_ndcg = sum(total_ndcg_list) / len(total_ndcg_list)

    return total_prec, total_ndcg