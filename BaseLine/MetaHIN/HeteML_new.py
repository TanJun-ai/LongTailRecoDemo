

import numpy as np
import torch
from torch.nn import functional as F
from MetaLearner_new import MetapathLearner, MetaLearner
from EmbeddingInitializer import User, Item
from torch.autograd import Variable
from Evaluation import add_metric


class HML(torch.nn.Module):
    def __init__(self, config, model_name, device):
        super(HML, self).__init__()
        self.config = config
        self.device = device
        self.model_name = model_name
        '''item 和 user 的embedding形式'''
        self.item_emb = Item(config)
        self.user_emb = User(config)

        self.mp_learner = MetapathLearner(config)
        self.meta_learner = MetaLearner(config)
        print("========================MetaLearner======================")
        print(self.meta_learner)
        print("=========================================================")

        self.mp_lr = config['mp_lr']
        self.local_lr = config['local_lr']
        self.emb_dim = self.config['embedding_dim']


        self.ml_weight_len = len(self.meta_learner.update_parameters())
        self.ml_weight_name = list(self.meta_learner.update_parameters().keys())
        self.mp_weight_len = len(self.mp_learner.update_parameters())
        self.mp_weight_name = list(self.mp_learner.update_parameters().keys())

        self.transformer_liners = self.transform_mp2task()

        self.meta_optimizer = torch.optim.Adam(self.parameters(), lr=config['lr'])

    def transform_mp2task(self):
        liners = {}
        ml_parameters = self.meta_learner.update_parameters()
        output_dim_of_mp = 16  # movielens: lr=0.001, avg mp, 0.8081
        for w in self.ml_weight_name:
            liners[w.replace('.', '-')] = torch.nn.Linear(output_dim_of_mp, np.prod(ml_parameters[w].shape))
        return torch.nn.ModuleDict(liners)

    def forward(self, support_user_emb, support_item_emb, support_set_y, support_mp_user_emb, vars_dict=None):

        if vars_dict is None:
            vars_dict = self.meta_learner.update_parameters()

        support_set_y_pred = self.meta_learner(support_user_emb, support_item_emb, support_mp_user_emb, vars_dict)
        loss = F.mse_loss(support_set_y_pred, support_set_y)
        grad = torch.autograd.grad(loss, vars_dict.values(), create_graph=True)

        fast_weights = {}
        for i, w in enumerate(vars_dict.keys()):
            fast_weights[w] = vars_dict[w] - self.local_lr * grad[i]

        for idx in range(1, self.config['local_update']):  # for the current task, locally update
            support_set_y_pred = self.meta_learner(support_user_emb, support_item_emb, support_mp_user_emb, vars_dict=fast_weights)
            loss = F.mse_loss(support_set_y_pred, support_set_y)  # calculate loss on support set
            grad = torch.autograd.grad(loss, fast_weights.values(),
                                       create_graph=True)  # calculate gradients w.r.t. model parameters

            for i, w in enumerate(fast_weights.keys()):
                fast_weights[w] = fast_weights[w] - self.local_lr * grad[i]

        return fast_weights

    def mp_update(self, support_set_x, support_set_y, support_set_mps, query_set_x, query_set_y, query_set_mps):
        """
        Mete-update the parameters of MetaPathLearner, AggLearner and MetaLearner.
        """
        # each mp
        support_mp_enhanced_user_emb_s, query_mp_enhanced_user_emb_s = [], []

        mp_initial_weights = self.mp_learner.update_parameters()
        ml_initial_weights = self.meta_learner.update_parameters()
        '''这里将one-hot编码转变为embedding，在lastfm-20中，item_fea_len=3846，user_fea_len=1872，
        转成embedding后统一为16维'''
        item_x = Variable(support_set_x[:, self.config['user_dim']:], requires_grad=False).float()
        user_x = Variable(support_set_x[:, 0:self.config['user_dim']], requires_grad=False).float()
        item_q = Variable(query_set_x[:, self.config['user_dim']:], requires_grad=False).float()
        user_q = Variable(query_set_x[:, 0:self.config['user_dim']], requires_grad=False).float()

        '''这里交换一下，因为item作为主键id不变，user作为交互的id'''
        support_user_emb = self.item_emb(item_x)
        support_item_emb = self.user_emb(user_x)
        query_user_emb = self.item_emb(item_q)
        query_item_emb = self.user_emb(user_q)

        '''首先要获取embedding层，包括support_user_emb，support_item_emb，support_mp_enhanced_user_emb'''
        support_set_mp = support_set_mps
        query_set_mp = query_set_mps

        support_neighs_emb = self.user_emb(torch.cat(support_set_mp).to(self.device))
        support_index_list = list(map(lambda _: _.shape[0], support_set_mp))
        query_neighs_emb = self.user_emb(torch.cat(query_set_mp).to(self.device))
        query_index_list = list(map(lambda _: _.shape[0], query_set_mp))

        support_mp_enhanced_user_emb = self.mp_learner(
            support_user_emb, support_item_emb,  support_neighs_emb, support_index_list)

        '''开始预测'''
        support_set_y_pred = self.meta_learner(support_user_emb, support_item_emb, support_mp_enhanced_user_emb)
        loss = F.mse_loss(support_set_y_pred, support_set_y)
        grad = torch.autograd.grad(loss, mp_initial_weights.values(), create_graph=True)

        fast_weights = {}
        for i in range(self.mp_weight_len):
            weight_name = self.mp_weight_name[i]
            fast_weights[weight_name] = mp_initial_weights[weight_name] - self.mp_lr * grad[i]

        for idx in range(1, self.config['mp_update']):
            support_mp_enhanced_user_emb = self.mp_learner(
                support_user_emb, support_item_emb, support_neighs_emb, support_index_list, vars_dict=fast_weights)
            support_set_y_pred = self.meta_learner(support_user_emb, support_item_emb, support_mp_enhanced_user_emb)
            loss = F.mse_loss(support_set_y_pred, support_set_y)
            grad = torch.autograd.grad(loss, fast_weights.values(), create_graph=True)

            for i in range(self.mp_weight_len):
                weight_name = self.mp_weight_name[i]
                fast_weights[weight_name] = fast_weights[weight_name] - self.mp_lr * grad[i]

        support_mp_enhanced_user_emb = self.mp_learner(support_user_emb, support_item_emb, support_neighs_emb, support_index_list, vars_dict=fast_weights)
        support_mp_enhanced_user_emb_s.append(support_mp_enhanced_user_emb)
        query_mp_enhanced_user_emb = self.mp_learner(query_user_emb, query_item_emb, query_neighs_emb, query_index_list, vars_dict=fast_weights)
        query_mp_enhanced_user_emb_s.append(query_mp_enhanced_user_emb)

        f_fast_weights = {}
        for w, liner in self.transformer_liners.items():
            w = w.replace('-', '.')
            f_fast_weights[w] = ml_initial_weights[w] * \
                                torch.sigmoid(liner(support_mp_enhanced_user_emb.mean(0))).view(ml_initial_weights[w].shape)
        # f_fast_weights = None
        # # the current mp ---> task update
        mp_task_fast_weights = self.forward(support_user_emb, support_item_emb, support_set_y,
                                            support_mp_enhanced_user_emb, vars_dict=f_fast_weights)
        mp_task_fast_weights_s = mp_task_fast_weights

        '''traing-set中的query数据集，用来进行测试来调整参数，这里要计算损失函数q_loss'''
        query_set_y_pred = self.meta_learner(query_user_emb, query_item_emb, query_mp_enhanced_user_emb,
                                             vars_dict=mp_task_fast_weights)
        q_loss = F.mse_loss(query_set_y_pred, query_set_y)
        mp_task_loss_s = q_loss.data  # movielens: 0.8126 dbook 0.6084

        '''[mp_task_loss_s] = [tensor(0.5125)]'''
        mp_att = F.softmax(-torch.stack([mp_task_loss_s]), dim=0)  # movielens: 0.80781 lr0.001
        agg_task_fast_weights = self.aggregator(mp_task_fast_weights_s, mp_att)
        agg_mp_emb = torch.stack(query_mp_enhanced_user_emb_s, 1)
        query_agg_enhanced_user_emb = torch.sum(agg_mp_emb * mp_att.unsqueeze(1), 1)

        '''预测分数并进行指标计算'''
        query_y_pred = self.meta_learner(query_user_emb, query_item_emb, query_agg_enhanced_user_emb, vars_dict=agg_task_fast_weights)
        test_output_list, query_recom_list = query_y_pred.view(-1).sort(descending=True)
        test_prec10, test_ndcg10 = add_metric(query_recom_list, query_set_y, topn=10)
        test_prec8, test_ndcg8 = add_metric(query_recom_list, query_set_y, topn=8)
        test_prec5, test_ndcg5 = add_metric(query_recom_list, query_set_y, topn=5)
        loss = F.mse_loss(query_y_pred, query_set_y)

        return loss, test_prec10, test_ndcg10, test_prec8, test_ndcg8, test_prec5, test_ndcg5

    def global_update(self, support_xs, support_ys, support_mps, query_xs, query_ys, query_mps, device):

        batch_sz = len(support_xs)  # batch_sz=32
        loss_s = []
        test_prec10 = []
        test_ndcg10 = []
        test_prec8 = []
        test_ndcg8 = []
        test_prec5 = []
        test_ndcg5 = []

        '''对每一批样本batch_sz=32进行训练'''
        for i in range(batch_sz):  # each task in a batch
            support_mp = support_mps
            query_mp = query_mps
            _loss, _test_prec10, _test_ndcg10, _test_prec8, _test_ndcg8, _test_prec5, _test_ndcg5 = self.mp_update(support_xs[i].to(device), support_ys[i].to(device), support_mp[i],
                                                         query_xs[i].to(device), query_ys[i].to(device), query_mp[i])

            loss_s.append(_loss)
            test_prec10.append(_test_prec10)
            test_ndcg10.append(_test_ndcg10)
            test_prec8.append(_test_prec8)
            test_ndcg8.append(_test_ndcg8)
            test_prec5.append(_test_prec5)
            test_ndcg5.append(_test_ndcg5)

        '''外层更新，meta-update'''
        loss = torch.stack(loss_s).mean(0)
        prec10 = sum(test_prec10)/len(test_prec10)
        ndcg10 = sum(test_ndcg10)/len(test_ndcg10)
        prec8 = sum(test_prec8) / len(test_prec8)
        ndcg8 = sum(test_ndcg8) / len(test_ndcg8)
        prec5 = sum(test_prec5) / len(test_prec5)
        ndcg5 = sum(test_ndcg5) / len(test_ndcg5)
        # prec = np.mean(test_prec)
        # ndcg = np.mean(test_ndcg)

        self.meta_optimizer.zero_grad()
        loss.backward()
        self.meta_optimizer.step()

        return loss, prec10, ndcg10, prec8, ndcg8, prec5, ndcg5

    def evaluation(self, support_x, support_y, support_mp, query_x, query_y, query_mp, device):

        loss, test_prec10, test_ndcg10, test_prec8, test_ndcg8, test_prec5, test_ndcg5, = self.mp_update(support_x.to(device), support_y.to(device), support_mp,
                                              query_x.to(device), query_y.to(device), query_mp)

        return loss, test_prec10, test_ndcg10, test_prec8, test_ndcg8, test_prec5, test_ndcg5

    def aggregator(self, task_weights_s, att):
        for idx, mp in enumerate(self.config['mp']):
            if idx == 0:
                att_task_weights = dict({k: v * att[idx] for k, v in task_weights_s.items()})
                continue
            tmp_att_task_weights = dict({k: v * att[idx] for k, v in task_weights_s.items()})
            att_task_weights = dict(zip(att_task_weights.keys(),
                                        list(map(lambda x: x[0] + x[1],
                                                 zip(att_task_weights.values(), tmp_att_task_weights.values())))))

        return att_task_weights



