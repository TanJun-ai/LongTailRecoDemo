
import math
import random
import torch
import numpy as np
from time import time

from BaseLine.MetaKG.utility.testing import model_testing
from BaseLine.MetaKG.utility.utils import LoaderDataset
from utility.parser_Metakg import parse_args
from BaseLine.MetaKG.data_loader import load_data
from model.MetaKG import Recommender

from utility.scheduler import Scheduler
from tqdm import tqdm

n_users = 0
n_items = 0
n_entities = 0
n_nodes = 0
n_relations = 0
sample_num = 10

def get_feed_dict(train_entity_pairs, start, end, train_user_set):
    def negative_sampling(user_item, train_user_set):
        neg_items = []
        for user, _ in user_item.cpu().numpy():
            user = int(user)
            while True:
                neg_item = np.random.randint(low=0, high=n_items, size=1)[0]
                if neg_item not in train_user_set[user]:
                    break
            neg_items.append(neg_item)
        return neg_items

    feed_dict = {}
    entity_pairs = train_entity_pairs[start:end].to(device)
    feed_dict['users'] = entity_pairs[:, 0]
    feed_dict['pos_items'] = entity_pairs[:, 1]
    feed_dict['neg_items'] = torch.LongTensor(negative_sampling(entity_pairs, train_user_set)).to(device)
    return feed_dict

def get_feed_dict_meta(support_user_set):
    support_meta_set = []
    for key, val in support_user_set.items():
        feed_dict = []
        user = [int(key)] * sample_num
        if len(val) != sample_num:
            pos_item = np.random.choice(list(val), sample_num, replace=True)
        else:
            pos_item = val

        neg_item = []
        while True:
            tmp = np.random.randint(low=0, high=n_items, size=1)[0]
            if tmp not in val:
                neg_item.append(tmp)
            if len(neg_item) == sample_num:
                break
        feed_dict.append(np.array(user))
        feed_dict.append(np.array(list(pos_item)))
        feed_dict.append(np.array(neg_item))
        support_meta_set.append(feed_dict)

    return np.array(support_meta_set)  # [n_user, 3, 10]

def get_feed_kg(kg_graph):
    triplet_num = len(kg_graph)
    pos_hrt_id = np.random.randint(low=0, high=triplet_num, size=args.batch_size * sample_num)
    pos_hrt = kg_graph[pos_hrt_id]
    neg_t = np.random.randint(low=0, high=n_entities, size=args.batch_size*sample_num)

    return torch.LongTensor(pos_hrt[:, 0]).to(device), torch.LongTensor(pos_hrt[:, 1]).to(device), torch.LongTensor(pos_hrt[:, 2]).to(device), torch.LongTensor(neg_t).to(device)

def convert_to_sparse_tensor(X):
    coo = X.tocoo()
    i = torch.LongTensor([coo.row, coo.col])
    v = torch.from_numpy(coo.data).float()
    return torch.sparse.FloatTensor(i, v, coo.shape).to(device)

def get_net_parameter_dict(params):
    param_dict = dict()
    indexes = []
    for i, (name, param) in enumerate(params):
        if param.requires_grad:
            param_dict[name] = param.to(device)
            indexes.append(i)

    return param_dict, indexes

def update_moving_avg(mavg, reward, count):
    return mavg + (reward.item() - mavg) / (count + 1)

if __name__ == '__main__':
    """fix the random seed"""
    seed = 2022
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    """read args"""
    global args, device
    args = parse_args()
    GPU = torch.cuda.is_available()
    device = torch.device('cuda:0' if GPU else 'cpu')
    args.mdevice = device

    item_size = 14  # 7*2

    """=========================build dataset training-set-start==============================="""
    ''''user_dict是字典类型，里面包括两个字段'train_user_set','test_user_set',
    分别用来训练和测试，n_params, graph, mat_list是构建知识图谱的关键信息，固定了.格式如下
    {'test_user_set':defaultdict(<class 'list'>, {3: [352, 736, 386, 325, 304, 338, 339, 343, 4539, 702],...})}'''
    train_supp_path = args.dataset_path + 'movielens-1m-200-800/training/top_supp_35_pos.txt'
    train_query_path = args.dataset_path + 'movielens-1m-200-800/training/top_query_7_pos.txt'
    train_cf, test_cf, user_dict, n_params, graph, mat_list = load_data(args, train_supp_path, train_query_path)
    adj_mat_list, mean_mat_list = mat_list
    args.n_item = n_params['n_users']
    args.m_user = n_params['n_items']

    """=========================testing-set-start==============================="""
    test_supp_path = args.dataset_path + 'movielens-1m-200-800/testing/tail_supp_7_pos.txt'
    test_query_path = args.dataset_path + 'movielens-1m-200-800/testing/tail_query_7_pos.txt'
    cold_train_cf, cold_test_cf, cold_user_dict, cold_n_params, cold_graph, cold_mat_list = load_data(args, test_supp_path, test_query_path)
    cold_adj_mat_list, cold_mean_mat_list = cold_mat_list
    test_dataset = LoaderDataset(args, test_query_path)
    """=========================testing-set-end================================="""

    kg_graph = np.array(list(graph.edges))  # [-1, 3]
    n_users = n_params['n_users']
    n_items = n_params['n_items']
    n_entities = n_params['n_entities']
    n_relations = n_params['n_relations']
    n_nodes = n_params['n_nodes']

    """cf data"""
    cold_train_cf_pairs = torch.LongTensor(np.array([[cf[0], cf[1]] for cf in cold_train_cf], np.int32))
    
    """use pretrain data,default=False"""
    if args.use_pretrain:
        print("==============use pretrain data===============")
        pre_path = args.data_path + 'pretrain/{}/mf.npz'.format(args.dataset)
        pre_data = np.load(pre_path)
        user_pre_embed = torch.tensor(pre_data['user_embed'])
        item_pre_embed = torch.tensor(pre_data['item_embed'])
    else:
        user_pre_embed = None
        item_pre_embed = None

    print("==========================MetaKG-model==========================")
    model = Recommender(n_params, args, graph, user_pre_embed, item_pre_embed).to(device)
    names_weights_copy, indexes = get_net_parameter_dict(model.named_parameters())
    scheduler = Scheduler(len(names_weights_copy), args, grad_indexes=indexes).to(device)
    print(model)
    print("----------------------------------------------------------------")

    """define optimizer"""
    optimizer = torch.optim.Adam(model.parameters(), lr=args.meta_update_lr)
    scheduler_optimizer = torch.optim.Adam(scheduler.parameters(), lr=args.scheduler_lr)

    """===============prepare feed data training-set============"""
    support_meta_set = get_feed_dict_meta(user_dict['train_user_set'])
    query_meta_set = get_feed_dict_meta(user_dict['test_user_set'])

    # shuffle
    index = np.arange(len(support_meta_set))
    np.random.shuffle(index)
    support_meta_set = support_meta_set[index]
    query_meta_set = query_meta_set[index]

    '''use_meta_model=false'''
    if args.use_meta_model:
        '''这里要说明使用cpu加载模型权重，否则默认是gpu，map_location=cpu'''
        model.load_state_dict(torch.load('./model_para/meta_model_{}.ckpt'.format(args.dataset), map_location='cpu'))
    else:
        print("start meta training ......")
        """meta training"""
        interact_mat = convert_to_sparse_tensor(mean_mat_list)
        model.interact_mat = interact_mat
        moving_avg_reward = 0

        model.train()
        iter_num = math.ceil(len(support_meta_set) / args.batch_size)
        train_s_t = time()
        for s in tqdm(range(iter_num)):
            batch_support = torch.LongTensor(support_meta_set[s * args.batch_size:(s + 1) * args.batch_size]).to(device)
            batch_query = torch.LongTensor(query_meta_set[s * args.batch_size:(s + 1) * args.batch_size]).to(device)

            pt = int(s / iter_num * 100)
            if len(batch_support) > args.meta_batch_size:
                task_losses, weight_meta_batch = scheduler.get_weight(batch_support, batch_query, model, pt)
                torch.cuda.empty_cache()
                task_prob = torch.softmax(weight_meta_batch.reshape(-1), dim=-1)
                selected_tasks_idx = scheduler.sample_task(task_prob, args.meta_batch_size)
                batch_support = batch_support[selected_tasks_idx]
                batch_query = batch_query[selected_tasks_idx]

            selected_losses = scheduler.compute_loss(batch_support, batch_query, model)
            meta_batch_loss = torch.mean(selected_losses)

            """KG loss"""
            h, r, pos_t, neg_t = get_feed_kg(kg_graph)
            kg_loss = model.forward_kg(h, r, pos_t, neg_t)
            batch_loss = kg_loss + meta_batch_loss

            """update scheduler"""
            loss_scheduler = 0
            for idx in selected_tasks_idx:
                loss_scheduler += scheduler.m.log_prob(idx.to(device))
            reward = meta_batch_loss
            loss_scheduler *= (reward - moving_avg_reward)
            moving_avg_reward = update_moving_avg(moving_avg_reward, reward, s)

            scheduler_optimizer.zero_grad()
            loss_scheduler.backward(retain_graph=True)
            scheduler_optimizer.step()

            """update network"""
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            torch.cuda.empty_cache()
        if args.save:
            torch.save(model.state_dict(), args.out_dir + 'meta_model_' + args.dataset + '.ckpt')
        train_e_t = time()
        print('meta_training_time: ', train_e_t-train_s_t)
    

    """===========================fine tune==================================="""
    # adaption u_i_interaction
    cold_interact_mat = convert_to_sparse_tensor(cold_mean_mat_list)
    model.interact_mat = cold_interact_mat
    # reset lr
    for g in optimizer.param_groups:
        g['lr'] = args.lr

    cur_best_pre_0 = 0
    stopping_step = 0
    should_stop = False

    """======================prepare feed data testing-set===================="""
    """meta-learning 之后就开始微调，即拿获得学习能力的模型再进行微调训练测试"""
    print("start fine tune......")
    max_prec = 0.
    max_ndcg = 0.
    for epoch in range(args.epoch):

        # shuffle training data
        index = np.arange(len(cold_train_cf))
        np.random.shuffle(index)
        cold_train_cf_pairs = cold_train_cf_pairs[index]
        '''=====模型训练testing-set中的support-set进行训练====='''
        model.train()
        loss = 0
        iter_num = math.ceil(len(cold_train_cf) / args.fine_tune_batch_size)
        train_s_t = time()
        for s in range(iter_num):
            batch = get_feed_dict(cold_train_cf_pairs,
                                  s*args.fine_tune_batch_size, (s+1) * args.fine_tune_batch_size,
                                  cold_user_dict['train_user_set'])

            batch_loss = model(batch, is_apapt=True)

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            loss += batch_loss.item()
        train_e_t = time()
        print("epoch:{}\t\ttrain_loss:{:.4f}\t\ttime:{:.4f}".format(epoch, loss, train_e_t-train_s_t))

        '''=============模型训练testing-set中的query-set进行测试================='''
        total_prec, total_ndcg = model_testing(args, test_dataset, model, item_size)
        if total_prec > max_prec:
            max_prec = total_prec
        if total_ndcg.item() > max_ndcg:
            max_ndcg = total_ndcg.item()
        if epoch % 10 == 0:
            print("----------------------query-testing---------------------")
            print("TOP-10: query_prec:{:.4f}\t\tquery_ndcg:{:.4f}".format(total_prec, total_ndcg.item()))
            print("max_prec:{:.4f}\t\tmax_ndcg:{:.4f}".format(max_prec, max_ndcg))
            print("-------------------------------------------------------")