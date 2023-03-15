
import random
from time import time
import torch
from testing import testing

'''用于训练模型,train_dataset=sup_x, sup_y, query_x, query_y的zip集合，长度为128x4'''
def training_melu(trainer, train_dataset, test_dataset, batch_size, num_epoch, model_save=True, model_filename=None):
    training_set_size = len(train_dataset)  # 128
    max_prec = 0.
    max_ndcg = 0.
    for epoch in range(num_epoch):  # num_epoch=100

        random.shuffle(train_dataset)   # 打乱train_dataset的顺序
        num_batch = int(training_set_size / batch_size)  # 128/batch_size=4
        a, b, c, d = zip(*train_dataset)

        trainer.train()  # 模型训练，torch.nn.Module自带的train()函数

        train_loss_list = []

        for i in range(num_batch):  # 4次循环
            try:
                # 分4个循环，每个循环训练batch_size=32个样本,拿到每个批次里的sup_x, sup_y, query_x, query_y的数据
                # [tensor([[0, 0, 0,  ..., 0, 0, 0],...]，每个样本的形式为一个user对应多个item
                supp_xs = list(a[batch_size*i:batch_size*(i+1)])  # 32个样本（user）的向量
                supp_ys = list(b[batch_size*i:batch_size*(i+1)])
                query_xs = list(c[batch_size*i:batch_size*(i+1)])
                query_ys = list(d[batch_size*i:batch_size*(i+1)])
            except IndexError:
                continue
            '''模型中的全局更新,返回训练损失函数值，和
            [[[0.13915673 0.14664052 0.14984488 0.13602175 0.14277859 0.14372464 0.14183296]]...]'''
            train_loss = trainer.global_update(supp_xs, supp_ys, query_xs, query_ys)
            train_loss_list.append(train_loss)

        loss = sum(train_loss_list)/len(train_loss_list)

        query_test_loss, query_test_prec, query_test_ndcg = testing(trainer, test_dataset)
        if query_test_prec > max_prec:
            max_prec = query_test_prec
        if query_test_ndcg > max_ndcg:
            max_ndcg = query_test_ndcg

        if epoch % 10 == 0:
            print("-------------query-testing------------------")
            print("TOP-"+str(trainer.args.top_k)+": query_loss:{:.4f}\t\tquery_prec:{:.4f}\t\tquery_ndcg:{:.4f}".format(
                query_test_loss.item(), query_test_prec, query_test_ndcg.item()))
            print("max_prec:{:.4f}\t\tmax_ndcg:{:.4f}".format(max_prec, max_ndcg))
            print("-------------------------------------------")

        print("epoch:{}\t\ttrain_loss:{:.4f}".format(epoch, loss.item()))



