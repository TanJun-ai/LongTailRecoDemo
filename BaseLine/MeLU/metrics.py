
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

'''
ranked_list=tensor([ 9,  2,  7,  6, 12,  4, 13,  0, 11, 10,  1,  5,  8,  3])
ground_truth=tensor([1., 0., 1., 0., 0., 0., 0., 1., 1., 0., 1., 1., 0., 0.])
然后我们取前top—10个item进行判断，对于precision指标，首先取正样本。
所以下面的 t[:10]=[1.,1.,1.,1.,1.,1.,0.,0.,0.,0.],
ranked_list的top-10为tensor([ 9,  2,  7,  6, 12,  4, 13,  0, 11, 10]),ground_truth[9]==0,然后查看是否在t中，
如果在，说明命中了，将t中的命中的0去掉，命中数+1。不太认同这个计算方法'''
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

