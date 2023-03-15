
import torch

class Item(torch.nn.Module):

    def __init__(self, args):
        super(Item, self).__init__()
        self.args = args
        self.num_item = args.item
        self.embedding_dim = args.embedding_dim

        '''one-hot向量展平成自定义的向量维度，比如这里的item的one-hot向量维度是self.num_item=3846，
        然后经过展平输出的维度为self.embedding_dim * 4=128'''
        self.embedding_item = torch.nn.Linear(
            in_features=self.num_item,
            out_features=self.embedding_dim * 4,
            bias=True
        )

    def forward(self, x):

        item_embedding = self.embedding_item(x).to(self.args.mdevice)
        return item_embedding

class User(torch.nn.Module):
    def __init__(self, args):
        super(User, self).__init__()
        self.args = args
        self.num_user = args.user
        self.embedding_dim = args.embedding_dim

        self.embedding_user = torch.nn.Linear(
            in_features=self.num_user,
            out_features=self.embedding_dim * 4,
            bias=True
        )

    def forward(self, x):

        user_embedding = self.embedding_user(x).to(self.args.mdevice)

        return user_embedding



