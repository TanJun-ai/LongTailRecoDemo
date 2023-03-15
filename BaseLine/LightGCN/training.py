
from time import time
import torch
from torch import optim
import utils

class BPRLossTraining:
    def __init__(self, recmodel, args):
        self.model = recmodel
        self.weight_decay = args.decay
        self.local_lr = args.local_lr
        self.opt = optim.Adam(self.model.parameters(), lr=self.local_lr)
    '''训练每一批数据，并进行参数的更新'''
    def batch_traning(self, items, pos, neg):
        loss, reg_loss = self.model.bpr_loss(items, pos, neg)
        # print("=======loss======")
        # print(loss)
        reg_loss = reg_loss*self.weight_decay
        loss = loss + reg_loss

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return loss.cpu().item()

'''对所有数据进行一次训练'''
def model_training(args, dataset, recommend_model, loss_class):
    start = time()
    Recmodel = recommend_model
    Recmodel.train()
    bpr_loss = loss_class

    '''均匀抽样,得到S=[[1269 3706 4989] ...]'''
    S = utils.UniformSample(dataset)

    items_id = torch.Tensor(S[:, 0]).long()
    pos_users_id = torch.Tensor(S[:, 1]).long()
    neg_items_id = torch.Tensor(S[:, 2]).long()

    items_id = items_id.to(args.mdevice)
    pos_users_id = pos_users_id.to(args.mdevice)
    neg_items_id = neg_items_id.to(args.mdevice)
    '''utils.shuffle用于随机打乱顺序，但对应关系不变'''
    items_id, pos_users_id, neg_items_id = utils.shuffle(items_id, pos_users_id, neg_items_id)

    total_batch = len(items_id) // args.bpr_batch + 1  # args.bpr_batch=2048
    aver_loss = 0.
    '''将组合矩阵(items_id, pos_users_id, neg_items_id）分成batch_i批'''
    for (batch_i, (batch_items, batch_pos, batch_neg)) in \
            enumerate(utils.minibatch(items_id, pos_users_id, neg_items_id, batch_size=args.bpr_batch)):
        '''获取每个批次的loss'''
        batch_loss = bpr_loss.batch_traning(batch_items, batch_pos, batch_neg)
        aver_loss += batch_loss

    aver_loss = aver_loss / total_batch
    end = time()
    return f"loss:{aver_loss:.3f} spend_times:{(end-start):.2f}"

