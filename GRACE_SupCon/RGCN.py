import torch.nn as nn
import dgl.nn as dglnn
import torch
import torch.nn.functional as F


class Rela_Attention(nn.Module):################################
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


class StochasticTwoLayerRGCN(nn.Module):
    def __init__(self, in_feat, hidden_feat, out_feat, rel_names, feat_dim):
        super().__init__()
        self.conv1 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(in_feat, hidden_feat, norm='right')
            for rel in rel_names
        })
        self.conv2 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(hidden_feat, out_feat, norm='right')
            for rel in rel_names
        })

        self.fc1 = torch.nn.Linear(out_feat, out_feat)
        self.fc2 = torch.nn.Linear(out_feat, out_feat)

        self.net = nn.Sequential(nn.Linear(out_feat, out_feat),
                        nn.ELU(),
                        nn.Linear(out_feat,  out_feat),
                        nn.ReLU(),
                        nn.Linear(out_feat, 2),
                        )
        self.net_cat = nn.Sequential(nn.Linear(out_feat+feat_dim, out_feat),
                        nn.ELU(),
                        nn.Linear(out_feat,  out_feat),
                        nn.ReLU(),
                        nn.Linear(out_feat, 2),
                        )
        self.dropout = nn.Dropout(p=0.3)
        self.dropout1 = nn.Dropout(p=0.3)

    def forward(self, blocks, x, category):
        """
        Parameters
        ---------
        blocks:
        x: torch.FloatTensor, [node_num, dim]
        """
        # if isinstance(blocks, list):
        #     x = self.conv1(blocks[0], x)
        #     # print('x: ', x)
        #     x = {k: F.relu(v) for k, v in x.items()}
        #     x = self.conv2(blocks[1], x)
        #     x = {k: F.relu(v) for k, v in x.items()}
        # else:
        #     x = self.conv1(blocks, x)
        #     x = {k: F.relu(v) for k, v in x.items()}
        #     x = self.conv2(blocks, x)
        #     x = {k: F.relu(v) for k, v in x.items()}
        # return x
        # dropout
        if isinstance(blocks, list):
            x = self.conv1(blocks[0], x)
            x = {k: F.relu(v) for k, v in x.items()}
            # x[category] = self.attention1(x[category])

            x = self.conv2(blocks[1], x)

            x = {k: F.relu(v) for k, v in x.items()}
            # x = {k: self.dropout(v) for k, v in x.items()}
            # x[category] = self.attention2(x[category])
        else:

            x = self.conv1(blocks, x)
            x = {k: F.relu(v) for k, v in x.items()}
            # x[category] = self.attention1(x[category])
            # x = {k: self.dropout(v) for k, v in x.items()}
            x = self.conv2(blocks, x)
            x = {k: F.relu(v) for k, v in x.items()}
            # x = {k: self.dropout(v) for k, v in x.items()}
            # x[category] = self.attention2(x[category])
        return x[category]

    def project(self, z: torch.Tensor) -> torch.Tensor:
        # 指定返回的类型为 torch.Tensor
        # z = self.dropout(z)
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def MLP(self, x):
        z = self.net(x)
        return z

    def MLP_cat(self, x):
        z = self.net_cat(x)
        return z


class StochasticTwoLayerRGCN_attention(nn.Module):
    def __init__(self, in_feat, hidden_feat, out_feat, rel_names):
        super().__init__()
        aggregate = 'stack'

        dic = {}
        for rel in rel_names:
            dic[rel] = dglnn.GraphConv(in_feat, hidden_feat, norm='right')
        self.conv1 = dglnn.HeteroGraphConv(dic, aggregate=aggregate)
        for rel in rel_names:
            dic[rel] = dglnn.GraphConv(in_feat, hidden_feat, norm='right')
        self.conv2 = dglnn.HeteroGraphConv(dic, aggregate=aggregate)

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
        if isinstance(blocks, list):
            x = self.conv1(blocks[0], x)
            x = {k: F.relu(v) for k, v in x.items()}
            x['review'] = self.attention1(x['review'])
            x = self.conv2(blocks[1], x)
            x = {k: F.relu(v) for k, v in x.items()}
            x['review'] = self.attention2(x['review'])
        else:
            x = self.conv1(blocks, x)
            x = {k: F.relu(v) for k, v in x.items()}
            x['review'] = self.attention1(x['review'])
            x = self.conv2(blocks, x)
            x = {k: F.relu(v) for k, v in x.items()}
            x['review'] = self.attention2(x['review'])
        return x

    def project(self, z: torch.Tensor) -> torch.Tensor:
        # 指定返回的类型为 torch.Tensor
        z = F.elu(self.fc1(z))
        return self.fc2(z)


class StochasticTwoLayerRGAT(nn.Module):
    def __init__(self,
                 g,
                 num_layers,
                 in_dim,
                 num_hidden,
                 out_dim,
                 rel_names,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual):
        super().__init__()
        self.g = g
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        if num_layers > 1:
            # input projection (no residual)
            self.gat_layers.append(dglnn.HeteroGraphConv({
            rel: dglnn.GATConv(in_dim, num_hidden, heads[0],
                feat_drop, attn_drop, negative_slope, False, self.activation)
                for rel in rel_names
            }))
            # hidden layers
            for l in range(1, num_layers - 1):
                # due to multi-head, the in_dim = num_hidden * num_heads
                self.gat_layers.append(dglnn.HeteroGraphConv({
            rel: dglnn.GATConv(
                    num_hidden * heads[l-1], num_hidden, heads[l],
                    feat_drop, attn_drop, negative_slope, residual, self.activation)
                    for rel in rel_names
            }))
            # output projection
            self.gat_layers.append(dglnn.HeteroGraphConv({
            rel: dglnn.GATConv(
                num_hidden * heads[-2], out_dim, heads[-1],
                feat_drop, attn_drop, negative_slope, residual, None)
                for rel in rel_names
        }))

        self.fc1 = torch.nn.Linear(out_dim, out_dim)
        self.fc2 = torch.nn.Linear(out_dim, out_dim)

    def forward(self, blocks, x):
        """
        Parameters
        ---------
        blocks:
        x: torch.FloatTensor, [node_num, dim]
        """
        if isinstance(blocks, list):
            x = self.gat_layers(blocks[0], x)
            # x = {k: F.relu(v) for k, v in x.items()}
            h= self.gat_layers(blocks[1], x)
            # h = {k: F.relu(v) for k, v in x.items()}
        else:
            h = x
            for l in range(self.num_layers):
                h = self.gat_layers[l](self.g, h)
                h = h.flatten(1) if l != self.num_layers - 1 else h.mean(1)
        return h

    def project(self, z: torch.Tensor) -> torch.Tensor:
        # 指定返回的类型为 torch.Tensor
        z = F.elu(self.fc1(z))
        return self.fc2(z)
