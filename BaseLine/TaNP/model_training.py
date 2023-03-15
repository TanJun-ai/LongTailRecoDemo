
import random
from eval import testing
from torch.autograd import Variable

def training(trainer, train_dataset, test_dataset, batch_size, num_epoch):
    training_set_size = len(train_dataset)
    max_prec10 = 0.
    max_ndcg10 = 0.
    max_prec7 = 0.
    max_ndcg7 = 0.
    max_prec5 = 0.
    max_ndcg5 = 0.
    for epoch in range(num_epoch):
        random.shuffle(train_dataset)
        num_batch = int(training_set_size / batch_size)
        a, b, c, d = zip(*train_dataset)
        trainer.train()
        all_C_distribs = []
        for i in range(num_batch):
            try:
                supp_xs = list(a[batch_size*i:batch_size*(i+1)])
                supp_ys = list(b[batch_size*i:batch_size*(i+1)])
                query_xs = list(c[batch_size*i:batch_size*(i+1)])
                query_ys = list(d[batch_size*i:batch_size*(i+1)])
            except IndexError:
                continue
            train_loss, batch_C_distribs = trainer.global_update(supp_xs, supp_ys, query_xs, query_ys)
            all_C_distribs.append(batch_C_distribs)

        P5, NDCG5, MAP5, P7, NDCG7, MAP7, P10, NDCG10, MAP10 = testing(trainer, test_dataset)
        if P10 > max_prec10:
            max_prec10 = P10
        if NDCG10 > max_ndcg10:
            max_ndcg10 =NDCG10
        if P7 > max_prec7:
            max_prec7 = P7
        if NDCG7 > max_ndcg7:
            max_ndcg7 =NDCG7
        if P5 > max_prec5:
            max_prec5 = P5
        if NDCG5 > max_ndcg5:
            max_ndcg5 =NDCG5
        print("Epoch:{}\tLoss:{:.4f}\t TOP-5 {:.4f}\t{:.4f}\t{:.4f}\t TOP-7: {:.4f}\t{:.4f}\t{:.4f}"
              "\tTOP-10: {:.4f}\t{:.4f}\t{:.4f}".format(epoch, train_loss, P5, NDCG5, MAP5, P7, NDCG7, MAP7, P10, NDCG10, MAP10))
        if epoch % 10 == 0:
            print("===========max_prec10 and max_ndcg10=============")
            print("max_prec:{:.4f}".format(max_prec10))
            print("max_ndcg:{:.4f}".format(max_ndcg10))
            print("===========max_prec8 and max_ndcg8=============")
            print("max_prec:{:.4f}".format(max_prec7))
            print("max_ndcg:{:.4f}".format(max_ndcg7))
            print("===========max_prec5 and max_ndcg5=============")
            print("max_prec:{:.4f}".format(max_prec5))
            print("max_ndcg:{:.4f}".format(max_ndcg5))
