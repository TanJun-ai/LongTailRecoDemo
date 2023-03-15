
import torch
from torch import nn

'''
LightGCN 模型'''
class BaseModel(nn.Module):

	def __init__(self, args, dataset):
		super(BaseModel, self).__init__()

		self.args = args
		self.dataset = dataset
		self.__init_weight()

	'''初始化模型结构和参数'''
	def __init_weight(self):
		self.num_items = self.dataset.n_item  # 3948
		self.num_users = self.dataset.m_user  # 6041

		self.latent_dim = self.args.latent_dim_rec  # 64
		self.n_layers = self.args.n_layer  # 3
		self.keep_prob = self.args.keep_prob  # 0.6
		'''对items和users的one-hot编码进行embedding，统一输出self.latent_dim=64维'''
		self.embedding_item = torch.nn.Embedding(
			num_embeddings=self.num_items, embedding_dim=self.latent_dim)
		self.embedding_user = torch.nn.Embedding(
			num_embeddings=self.num_users, embedding_dim=self.latent_dim)

		'''如果不使用预训练，0表示不用，则需要初始化normal_，如果使用预训练，则直接加载预训练好的参数'''
		if self.args.pretrain == 0:
			#             nn.init.xavier_uniform_(self.embedding_user.weight, gain=1)
			#             nn.init.xavier_uniform_(self.embedding_item.weight, gain=1)
			#             print('use xavier initilizer')
			# random normal init seems to be a better choice when lightGCN actually don't use any non-linear activation function
			nn.init.normal_(self.embedding_user.weight, std=0.1)
			nn.init.normal_(self.embedding_item.weight, std=0.1)
			print('use normal distribution initilizer')
		else:
			#  这里需要修改!!!
			self.embedding_user.weight.data.copy_(torch.from_numpy(self.config['user_emb']))
			self.embedding_item.weight.data.copy_(torch.from_numpy(self.config['item_emb']))
			print('use pretrain data')
		self.f = nn.Sigmoid()
		self.Graph = self.dataset.getSparseGraph()

		print(f"lgn is already to go(dropout:{self.args.dropout})")

	'''对模型的graph层进行dropout'''
	def _dropout(self, x, keep_prob):
		size = x.size()
		index = x.indices().t()
		values = x.values()
		random_index = torch.rand(len(values)) + keep_prob
		random_index = random_index.int().bool()
		index = index[random_index]
		values = values[random_index] / keep_prob
		g = torch.sparse.FloatTensor(index.t(), values, size)
		return g

	'''定义数据在graph层传播'''
	def computer_graph_embs(self):
		"""信息的传播方式
		propagate methods for lightGCN"""
		items_emb = self.embedding_item.weight
		users_emb = self.embedding_user.weight
		all_emb = torch.cat([items_emb, users_emb])
		embeddings = [all_emb]

		if self.args.dropout:
			print("use dropout for LightGCN")
			g_drop = self._dropout(self.Graph, self.keep_prob)
		else:
			g_drop = self.Graph

		'''定义了3层图卷积的传播机制，用embedding来表示图的结构,
		embeddings=[tensor([[ 0.0027,  0.1076,  0.1851,  ..., -0.0048,  0.0390, -0.1562],...])]，
		一开始embeddings只有一层，里面有9993个一维list，每一个一维list都有64维构成，后面经过3层图卷积的传播，
		得到的embeddings有4层（原来的1+新增的3），每层形状一样，只是参数不同，
		代表user和item的组合特征all_emb在3层图卷积中传播后得到的不同参数'''
		for layer in range(self.n_layers):
			all_emb = torch.sparse.mm(g_drop, all_emb)
			embeddings.append(all_emb)

		embeddings = torch.stack(embeddings, dim=1)
		light_out = torch.mean(embeddings, dim=1)
		items, users = torch.split(light_out, [self.num_items, self.num_users])

		return items, users

	'''获得三元组<item, user+, user->的embedding向量表示,传入的items为有交互记录的items_id，
	tensor([1963,  209,  486,  ..., 2827, 3575, 1418])，一批的大小为2048，表示将这2048个items_id用
	64维的embedding层表示'''
	def getEmbedding(self, items, pos_users, neg_users):
		all_items, all_users = self.computer_graph_embs()
		items_emb = all_items[items]
		pos_emb = all_users[pos_users]
		neg_emb = all_users[neg_users]
		items_emb_ego = self.embedding_item(items)
		pos_emb_ego = self.embedding_user(pos_users)
		neg_emb_ego = self.embedding_user(neg_users)
		return items_emb, pos_emb, neg_emb, items_emb_ego, pos_emb_ego, neg_emb_ego

	'''对照公式去理解，有两个损失函数，一个是正负样本的损失函数loss，一个是L2正则项损失函数reg_loss'''
	def bpr_loss(self, items, pos, neg):
		(items_emb, pos_emb, neg_emb, itemEmb0,  posEmb0, negEmb0) = self.getEmbedding(items.long(), pos.long(), neg.long())
		reg_loss = (1/2) * (itemEmb0.norm(2).pow(2) + posEmb0.norm(2).pow(2) + negEmb0.norm(2).pow(2)) / float(len(items))
		'''正样本得分,items_emb一批有2048个item_id，pos_emb一批有2048个user_id，
		所以得到2048个item对user相似度的分数'''
		pos_scores = torch.mul(items_emb, pos_emb)
		pos_scores = torch.sum(pos_scores, dim=1)

		'''负样本得分'''
		neg_scores = torch.mul(items_emb, neg_emb)
		neg_scores = torch.sum(neg_scores, dim=1)

		loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

		return loss, reg_loss

	'''获得item对应的所有users的推荐分数，用这个分数进行topk筛选，这个函数在Testing时用到，
	这是是获得一批100个item，每个item对应的6040个user的分数，然后使用topk函数取出分数前topk个user的下标，
	即user的id，使用的还是两个向量的内积来获得分数'''
	def getItemsRating(self, items, users):
		# all_items大小为3952，包含了所有的item；all_users为6040，包含了所有的user
		all_items, all_users = self.computer_graph_embs()

		items_emb = all_items[items.long()]  # 大小为items=（100，64）
		users_emb = all_users[users.long()]  # 形状users_emb.t()=（64，6040）
		rating = self.f(torch.matmul(items_emb, users_emb.t()))  # 形状（100,6040）
		self.get_user_emb = all_users
		return rating



