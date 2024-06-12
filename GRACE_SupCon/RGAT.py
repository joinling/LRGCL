import torch.nn as nn
import dgl.nn as dglnn
import torch
import torch.nn.functional as F
from dgl import apply_each


class Rela_Attention(nn.Module):
    def __init__(self, in_size, hidden_size=32):
        super(Rela_Attention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        # print( z.shape)
        w = self.project(z).mean(0)  # (M, 1)
        # print( w.shape)
        beta = torch.softmax(w, dim=0)  # (M, 1)
        beta = beta.expand((z.shape[0],) + beta.shape)  # (N, M, 1)
        # print(beta.shape)
        return (beta * z).sum(1)


class StochasticTwoLayerRGAT(nn.Module):
    def __init__(self, in_feat, hidden_feat, out_feat, rel_names, n_heads=1, n_class=2):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.HeteroGraphConv({
            rel: dglnn.GATConv(in_feat, hidden_feat//n_heads, n_heads)
            for rel in rel_names}))
        self.layers.append(dglnn.HeteroGraphConv({
            rel: dglnn.GATConv(hidden_feat, hidden_feat//n_heads, n_heads)
            for rel in rel_names}))

        self.dropout = nn.Dropout(0.3)
        self.linear = nn.Linear(hidden_feat, out_feat)

        self.fc1 = torch.nn.Linear(hidden_feat, out_feat)
        self.fc2 = torch.nn.Linear(out_feat, out_feat)

        self.net = nn.Sequential(nn.Linear(out_feat, out_feat),
                        nn.ReLU(),
                        nn.Linear(out_feat,  out_feat),
                        nn.ELU(),
                        nn.Linear(out_feat, n_class),
                        )

    def forward(self, blocks, x, category):
        h = x
        # mini-batch training
        if isinstance(blocks, list):
            for l, (layer, block) in enumerate(zip(self.layers, blocks)):
                h = layer(block, h)
                # One thing is that h might return tensors with zero rows if the number of dst nodes
                # of one node type is 0.  x.view(x.shape[0], -1) wouldn't work in this case.
                h = apply_each(h, lambda x: x.view(x.shape[0], x.shape[1] * x.shape[2]))
                if l != len(self.layers) - 1:
                    h = apply_each(h, F.relu)
                    h = apply_each(h, self.dropout)
        # full-batch training
        else:
            for l, layer in enumerate(self.layers):
                h = layer(blocks, h)
                # One thing is that h might return tensors with zero rows if the number of dst nodes
                # of one node type is 0.  x.view(x.shape[0], -1) wouldn't work in this case.
                h = apply_each(h, lambda x: x.view(x.shape[0], x.shape[1] * x.shape[2]))
                if l != len(self.layers) - 1:
                    h = apply_each(h, F.relu)
                    h = apply_each(h, self.dropout)

        return self.linear(h[category])

    def project(self, z: torch.Tensor) -> torch.Tensor:
        # 指定返回的类型为 torch.Tensor
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def MLP(self, x):
        z = self.net(x)
        return z

# 调不通
class StochasticTwoLayerRGAT_attention(nn.Module):
    def __init__(self, in_feat, hidden_feat, out_feat, rel_names, n_heads=2):
        super().__init__()
        aggregate = 'stack'

        self.layers = nn.ModuleList()
        self.layers.append(dglnn.HeteroGraphConv({
            rel: dglnn.GATConv(in_feat, hidden_feat//n_heads, n_heads)
            for rel in rel_names}, aggregate=aggregate))
        self.layers.append(dglnn.HeteroGraphConv({
            rel: dglnn.GATConv(hidden_feat, hidden_feat//n_heads, n_heads)
            for rel in rel_names}, aggregate=aggregate))

        self.dropout = nn.Dropout(0.3)
        self.linear = nn.Linear(hidden_feat, out_feat)

        self.attention1 = Rela_Attention(in_size=hidden_feat)
        self.attention2 = Rela_Attention(in_size=out_feat)

        self.fc1 = torch.nn.Linear(hidden_feat, out_feat)
        self.fc2 = torch.nn.Linear(out_feat, hidden_feat)

    def forward(self, blocks, x):
        """
        Parameters
        ---------
        blocks:
        x: torch.FloatTensor, [node_num, dim]
        """
        h = x
        if isinstance(blocks, list):
            for l, (layer, block) in enumerate(zip(self.layers, blocks)):
                h = layer(block, h)
                print('h: ', h)
                # One thing is that h might return tensors with zero rows if the number of dst nodes
                # of one node type is 0.  x.view(x.shape[0], -1) wouldn't work in this case.
                h = apply_each(h, lambda x: x.view(x.shape[0], x.shape[1] * x.shape[2]))
                if l != len(self.layers) - 1:
                    h = apply_each(h, F.relu)
                    h = apply_each(h, self.dropout)
                    h['review'] = self.attention1(h['review'])
                else:
                    h['review'] = self.attention2(h['review'])
        else:
            for l, layer in enumerate(self.layers):
                h = layer(blocks, h)
                # One thing is that h might return tensors with zero rows if the number of dst nodes
                # of one node type is 0.  x.view(x.shape[0], -1) wouldn't work in this case.
                h = apply_each(h, lambda x: x.view(x.shape[0], x.shape[1] * x.shape[2]))
                if l != len(self.layers) - 1:
                    h = apply_each(h, F.relu)
                    h = apply_each(h, self.dropout)
                    h['review'] = self.attention1(h['review'])
                else:
                    h['review'] = self.attention2(h['review'])

        return h

    def project(self, z: torch.Tensor) -> torch.Tensor:
        # 指定返回的类型为 torch.Tensor
        z = F.elu(self.fc1(z))
        return self.fc2(z)