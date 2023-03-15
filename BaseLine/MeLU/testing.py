
def testing(trainer, test_dataset):
    test_dataset_len = len(test_dataset)  # 只有66个长度，一次性测完

    minibatch_size = 1
    a, b, c, d = zip(*test_dataset)
    trainer.eval()
    total_loss_list = []
    total_ndcg_list = []
    total_prec_list = []

    for i in range(test_dataset_len):
        try:
            supp_xs = list(a[minibatch_size * i:minibatch_size * (i + 1)])
            supp_ys = list(b[minibatch_size * i:minibatch_size * (i + 1)])
            query_xs = list(c[minibatch_size * i:minibatch_size * (i + 1)])
            query_ys = list(d[minibatch_size * i:minibatch_size * (i + 1)])
        except IndexError:
            continue

        test_loss, test_prec, test_ndcg = trainer.test_rec(supp_xs, supp_ys, query_xs, query_ys)
        total_loss_list.append(test_loss)
        total_prec_list.append(test_prec)
        total_ndcg_list.append(test_ndcg)


    total_loss = sum(total_loss_list)/len(total_loss_list)
    total_prec = sum(total_prec_list) / len(total_prec_list)
    total_ndcg = sum(total_ndcg_list)/len(total_ndcg_list)


    return total_loss, total_prec, total_ndcg

