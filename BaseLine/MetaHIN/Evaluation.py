

import math


def NDCG(ranked_list, ground_truth, topn):
    dcg = 0
    idcg = IDCG(ground_truth, topn)

    for i in range(topn):
        id = ranked_list[i]
        dcg += ((2 ** ground_truth[id]) - 1)/math.log(i+2, 2)

    return dcg / idcg

def IDCG(ground_truth, topn):
    t = [a for a in ground_truth]
    t.sort(reverse=True)
    idcg = 0
    for i in range(topn):
        idcg += ((2**t[i]) - 1) / math.log(i+2, 2)
    return idcg

def precision(ranked_list, ground_truth, topn):
    t = [a for a in ground_truth]
    t.sort(reverse=True)
    t = t[:topn]
    hits = 0
    for i in range(topn):
        id = ranked_list[i]
        if ground_truth[id] in t:
            t.remove(ground_truth[id])
            hits += 1
    pre = hits/topn
    return pre

def add_metric(recommend_list, all_group_list, topn):

    ndcg = NDCG(recommend_list, all_group_list, topn)
    prec = precision(recommend_list, all_group_list, topn)

    return prec, ndcg







