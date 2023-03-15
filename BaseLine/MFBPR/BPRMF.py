
import torch
import torch.nn as nn


class BPRMF(nn.Module):

	def __init__(self, n_users, n_items, args):

		super(BPRMF, self).__init__()

		self.args = args
		self.n_users = n_users
		self.n_items = n_items
		self.emb_size = args.emb_size
		self.batch_size = args.batch_size
		self.decay = args.decay

		self.user_embeddings = nn.Embedding(self.n_users, embedding_dim=self.emb_size)
		self.item_embeddings = nn.Embedding(self.n_items, embedding_dim=self.emb_size)

		self._weight_init()

	'''初始化embedding层的权重'''
	def _weight_init(self):
		nn.init.normal_(self.user_embeddings.weight, std=0.1)
		nn.init.normal_(self.item_embeddings.weight, std=0.1)
		# nn.init.xavier_uniform_(self.user_embeddings.weight, gain=1)
		# nn.init.xavier_uniform_(self.item_embeddings.weight, gain=1)
		print('use unnormal distribution initilizer')

	def get_embedding(self, users, pos_items, neg_items):
		all_users = self.user_embeddings.weight.to(self.args.mdevice)
		all_items = self.item_embeddings.weight.to(self.args.mdevice)
		u_emb = all_users[users]
		pos_emb = all_items[pos_items]
		neg_emb = all_items[neg_items]
		u_emb_ego = self.user_embeddings(users)
		pos_emb_ego = self.item_embeddings(pos_items)
		neg_emb_ego = self.item_embeddings(neg_items)

		return u_emb, pos_emb, neg_emb, u_emb_ego, pos_emb_ego, neg_emb_ego


	def forward(self, user_emb, item_emb):

		score = torch.mul(user_emb, item_emb)
		return score