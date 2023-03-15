
import torch
from torch.autograd import Variable
from torch.nn import functional as F
from embeddings import Item, User
from metrics import add_metric

'''base-learner'''
class Base_Learner(torch.nn.Module):
    '''生成一个MLP网络'''
    def __init__(self, args):
        super(Base_Learner, self).__init__()
        self.args = args
        self.embedding_dim = args.embedding_dim  # 32
        self.fc1_in_dim = args.embedding_dim * 8  # 32*8
        self.fc2_in_dim = args.first_fc_hidden_dim  # 64
        self.fc2_out_dim = args.second_fc_hidden_dim * 2  # 64

        self.item_emb = Item(args)
        self.user_emb = User(args)
        self.fc1 = torch.nn.Linear(self.fc1_in_dim, self.fc2_in_dim)
        self.fc2 = torch.nn.Linear(self.fc2_in_dim, self.fc2_out_dim)

        self.linear_out = torch.nn.Linear(self.fc2_out_dim, 1)

    '''item和user的特征在网络中的传播'''
    def forward(self, x):
        '''这里没有打乱user的顺序，不知道打乱顺序会怎样'''
        item_x = Variable(x[:, 0:self.args.item].float(), requires_grad=True)
        user_x = Variable(x[:, self.args.item:].float(), requires_grad=True)
        '''item_emb的一个形状(20,128)，表示有20个item，每个item用128维表示；
        user_emb的一个形状（20,128），表示有1个user,用128维表示，这里表示一个user对应20个item，
        所以输入的user_emb=20都是一个user的表示形式，参数都是一样的'''
        item_emb = self.item_emb(item_x).to(self.args.mdevice)
        user_emb = self.user_emb(user_x).to(self.args.mdevice)
        '''x的长度为user对应的有交互item的数量，这里的support-set的固定为20，拼接是横向拼接，item在前。
        x[0]的长度必须等于self.fc1_in_dim的长度，这里固定为256，item_emb和user_emb的长度固定为128'''
        x = torch.cat((item_emb, user_emb), 1)
        f1 = self.fc1(x)
        fn1 = F.relu(f1)
        f2 = self.fc2(fn1)
        fn2 = F.relu(f2)

        '''经过卷积神经网络，得出预测的一个分数，即这个分数是关于这个user对这个item的评分'''
        return self.linear_out(fn2)

'''meta-learner'''
class MeLU(torch.nn.Module):

    def __init__(self, args, device):
        super(MeLU, self).__init__()

        self.device = device
        self.args = args
        self.local_lr = args.local_lr
        self.model = Base_Learner(self.args)  # 基础模型base_learner

        self.meta_optim = torch.optim.Adam(self.model.parameters(), lr=self.args.global_lr)
        self.local_optim = torch.optim.Adam(self.model.parameters(), lr=self.args.local_lr)

    '''全局更新，即元学习的更新.这里的训练方式，先是对一个批32个user进行训练，得出train_loss并进行参数更新操作，
    训练完support-set之后，才进行query-set的预测操作，也是对一个批32个user进行预测，得到test_loss，并进行更新，
    这样，就是两次更新过程，这会使得模型具有学习能力'''
    def global_update(self, support_set_xs, support_set_ys, query_set_xs, query_set_ys):

        batch_sz = len(support_set_xs)  # batch_sz=32
        losses_q = []
        train_loss = []

        num_local_update = 1
        '''对每个类（user）进行gpu训练，每一批有32个类,这里固定了support_set_xs[i] 的大小为20，query_set_xs[i]大小为15
        即support_set_xs[i]表示一个user，对应有20个有交互记录的item，是个二维tensor，里面的每个一维tensor表示为
        （item,user）tensor([0, 0, 0,  ..., 0, 0, 0])，注意item在前面，item,user都用one-hot向量表示了，
        item有3846个，user有1872个，故（item,user）前面的3846维表示item_embedding，后面的1872维表示user_embedding'''
        for i in range(batch_sz):
            support_set_xs[i] = support_set_xs[i].to(self.device)
            support_set_ys[i] = support_set_ys[i].to(self.device)

            '''局部参数的更新，使用Adam优化算法'''
            for idx in range(num_local_update):
                '''输出一个user对应的20个item的分数，这20个item是随机抽取的，包括正负样本'''
                support_set_y_pred = self.model(support_set_xs[i])
                loss = F.mse_loss(support_set_y_pred, support_set_ys[i].view(-1, 1))
                train_loss.append(loss)

        t_loss = torch.stack(train_loss).mean(0)
        self.local_optim.zero_grad()
        t_loss.backward()
        self.local_optim.step()

        '''local update，局部更新，得出预测的query_set的标签query_set_y_pred，然后与真实的标签query_set_ys[i]
        做对比，得出损失值，这里使用F.mse_loss损失函数四个集合support_set_xs, support_set_ys, 
        query_set_xs, query_set_ys都用在了局部更新的步骤上'''
        for i in range(batch_sz):
            query_set_xs[i] = query_set_xs[i].to(self.device)
            query_set_ys[i] = query_set_ys[i].to(self.device)

            query_set_y_pred = self.model(query_set_xs[i])
            loss_q = F.mse_loss(query_set_y_pred, query_set_ys[i].view(-1, 1))
            losses_q.append(loss_q)

        losses = torch.stack(losses_q).mean(0)
        self.meta_optim.zero_grad()  # 清空所有被优化过的Variable的梯度
        losses.backward()  # 误差的反向传播,这步必须执行。利用反向传播算法来更新整个模型的参数
        self.meta_optim.step()  # 更新所有的参数，使用的是Adam，

        return losses


    '''只预测一个user，所以batch_sz = 1'''
    def test_rec(self, support_set_xs, support_set_ys, query_set_xs, query_set_ys):
        batch_sz = 1
        num_local_update = 1
        train_loss = []

        for i in range(batch_sz):
            support_set_xs[i] = support_set_xs[i].to(self.device)
            support_set_ys[i] = support_set_ys[i].to(self.device)

            for idx in range(num_local_update):
                support_set_y_pred = self.model(support_set_xs[i])
                loss = F.mse_loss(support_set_y_pred, support_set_ys[i].view(-1, 1))
                train_loss.append(loss)
        '''常规更新'''
        supp_loss = torch.stack(train_loss).mean(0)
        self.local_optim.zero_grad()
        supp_loss.backward()
        self.local_optim.step()

        '''最后进行预测输出，这里不再需要更新模型参数'''
        for i in range(batch_sz):
            query_set_xs[i] = query_set_xs[i].to(self.device)
            query_set_ys[i] = query_set_ys[i].to(self.device)

            query_set_y_pred = self.model(query_set_xs[i])

            test_output_list, query_recom_list = query_set_y_pred.view(-1).sort(descending=True)
            test_prec, test_ndcg = add_metric(query_recom_list, query_set_ys[i], topn=self.args.top_k)
            loss_q = F.mse_loss(query_set_y_pred, query_set_ys[i].view(-1, 1))


            return loss_q, test_prec, test_ndcg
