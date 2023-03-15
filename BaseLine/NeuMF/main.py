
import os
import time
import argparse

import torch
import torch.nn as nn
import torch.optim as optim



import model
import config
import evaluate



parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
parser.add_argument("--dropout", type=float, default=0.0, help="dropout rate")
parser.add_argument("--batch_size", type=int, default=256, help="batch size for training")
parser.add_argument("--epochs", type=int, default=200, help="training epoches")
parser.add_argument("--top_k", type=int, default=10, help="compute metrics@top_k")
parser.add_argument("--factor_num", type=int, default=32, help="predictive factors numbers in the model")
parser.add_argument("--num_layers", type=int, default=3, help="number of layers in MLP model")
parser.add_argument("--num_ng", type=int, default=4, help="sample negative items for training")
parser.add_argument("--test_num_ng", type=int, default=99, help="sample part of negative items for testing")
parser.add_argument("--out", default=True, help="save model or not")
parser.add_argument("--gpu", type=str, default="0", help="gpu card ID")
args = parser.parse_args()

'''设置运行的cuda'''
GPU = torch.cuda.is_available()
mdevice = torch.device('cuda:0' if GPU else 'cpu')
args.mdevice = mdevice

print("==================training-set-start===========================")
main_path = '../../data/lastfm-20-100-400/'
supp_path = main_path + 'testing/tail_supp_14.txt'
supp_label_path = main_path + 'testing/tail_supp_14_label.txt'
training_set_size = 400
user_num = 1872
item_num = 3846
train_items_id_list = []
train_users_id_list = []
train_labels_list = []
for i in range(training_set_size):
    cache_list = []
    with open(supp_path) as f:
        for l in f.readlines():
            if len(l) > 0:
                l = l.strip('\n').split(' ')
                cache_list.append(l)
    items_id = []
    for j in range(len(cache_list[i])-1):
        items_id.append(int(cache_list[i][0]))
    users_id = []
    users_id.extend(int(i) for i in cache_list[i][1:])
    train_items_id_list.append(torch.Tensor(items_id).long())
    train_users_id_list.append(torch.Tensor(users_id).long())

for j in range(training_set_size):
    cache_list = []
    with open(supp_label_path) as f:
        for l in f.readlines():
            if len(l) > 0:
                l = l.strip('\n').split(' ')
                cache_list.append(l)
    labels_id = []
    labels_id.extend(float(i) for i in cache_list[j][1:])
    train_labels_list.append(torch.Tensor(labels_id).long())

print("training_supp_items:" + str(item_num))
print("training_supp_users:" + str(user_num))
print("=====================training-set-end=======================")

print("==================testing-set-start===========================")
query_path = main_path + 'testing/tail_query_14.txt'
query_label_path = main_path + 'testing/tail_query_14_label.txt'
testing_set_size = 400
test_items_id_list = []
test_users_id_list = []
test_labels_list = []
for i in range(testing_set_size):
    cache_list = []
    with open(query_path) as f:
        for l in f.readlines():
            if len(l) > 0:
                l = l.strip('\n').split(' ')
                cache_list.append(l)
    items_id = []
    for j in range(len(cache_list[i])-1):
        items_id.append(int(cache_list[i][0]))
    users_id = []
    users_id.extend(int(i) for i in cache_list[i][1:])
    test_items_id_list.append(torch.Tensor(items_id).long())
    test_users_id_list.append(torch.Tensor(users_id).long())

for j in range(testing_set_size):
    cache_list = []
    with open(query_label_path) as f:
        for l in f.readlines():
            if len(l) > 0:
                l = l.strip('\n').split(' ')
                cache_list.append(l)
    labels_id = []
    labels_id.extend(float(i) for i in cache_list[j][1:])
    test_labels_list.append(torch.Tensor(labels_id).long())

print("testing_query_size:" + str(testing_set_size))
print("=====================testing-set-end=======================")

if config.model == 'NeuMF-pre':
    assert os.path.exists(config.GMF_model_path), 'lack of GMF model'
    assert os.path.exists(config.MLP_model_path), 'lack of MLP model'
    GMF_model = torch.load(config.GMF_model_path)
    MLP_model = torch.load(config.MLP_model_path)
else:
    GMF_model = None
    MLP_model = None

# model = model.NCF(user_num, item_num, args.factor_num, args.num_layers, args.dropout, config.model, GMF_model, MLP_model)
model = model.NCF(item_num, user_num, args.factor_num, args.num_layers, args.dropout, config.model, GMF_model, MLP_model)
model.to(args.mdevice)
loss_function = nn.BCEWithLogitsLoss()
'''MeuMF模型使用SGD优化器'''
if config.model == 'NeuMF-pre':
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=0.0001)
else:
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0001)


########################### TRAINING #####################################
count, best_hr = 0, 0
max_ndcg = 0.
max_prec = 0.
for epoch in range(args.epochs):

    '''================training-start=================='''
    model.train()  # Enable dropout (if have).
    start_time = time.time()

    train_loss = []
    for j in range(training_set_size):
        train_items_id_list[j] = train_items_id_list[j].to(args.mdevice)
        train_users_id_list[j] = train_users_id_list[j].to(args.mdevice)
        train_labels_list[j] = train_labels_list[j].to(args.mdevice)
        model.zero_grad()
        prediction = model(train_items_id_list[j], train_users_id_list[j])

        loss = loss_function(prediction, train_labels_list[j].float())
        train_loss.append(loss)
        loss.backward()
        optimizer.step()

    t_loss = torch.stack(train_loss).mean(0)
    end_time = time.time()
    print(f'EPOCH[{epoch + 1}/{args.epochs}] Loss[{t_loss:.3f}] Time[{end_time - start_time:.3f}]')

    '''================testing-start=================='''
    model.eval()
    total_ndcg_list = []
    total_prec_list = []
    for i in range(testing_set_size):
        test_items_id_list[i] = test_items_id_list[i].to(args.mdevice)
        test_users_id_list[i] = test_users_id_list[i].to(args.mdevice)
        test_labels_list[i] = test_labels_list[i].to(args.mdevice)

        test_y_prediction = model(test_items_id_list[i], test_users_id_list[i])

        test_output_list, query_recom_list = test_y_prediction.view(-1).sort(descending=True)

        test_prec, test_ndcg = evaluate.add_metric(query_recom_list, test_labels_list[j], topn=10)
        total_prec_list.append(test_prec)
        total_ndcg_list.append(test_ndcg)

    total_prec = sum(total_prec_list) / len(total_prec_list)
    total_ndcg = sum(total_ndcg_list) / len(total_ndcg_list)
    if (total_prec > max_prec) and (total_ndcg.item() > max_ndcg):
        if not os.path.exists(config.model_path):
            os.mkdir(config.model_path)
        torch.save(model, '{}{}.pth'.format(config.model_path, config.model))
    if total_prec > max_prec:
        max_prec = total_prec
    if total_ndcg.item() > max_ndcg:
        max_ndcg = total_ndcg.item()
    if epoch % 10 == 0:
        print("----------------------query-testing---------------------")
        print("TOP-10: query_prec:{:.4f}\t\tquery_ndcg:{:.4f}".format(total_prec, total_ndcg.item()))
        print("max_prec:{:.4f}\t\tmax_ndcg:{:.4f}".format(max_prec, max_ndcg))
        print("-------------------------------------------------------")
