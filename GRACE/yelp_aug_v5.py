import scipy.io as scio
import dgl
import scipy.sparse as sp
import torch
import random
import numpy as np
from scipy.sparse import coo_array


def sparse_to_adj_tensor(sp_matrix):

    """
    Transfer sparse matrix to adjacency list
    :param sp_matrix: the sparse matrix
    :param filename: the filename of adjlist
    """
    # add self loop
    homo_adj = sp_matrix + sp.eye(sp_matrix.shape[0])
    # create adj_np_list
    edges = homo_adj.nonzero()
    src = torch.tensor(edges[0])
    dst = torch.tensor(edges[1])
    edges = torch.stack([src, dst], dim=0)
    # edges = np.array(edges)
    return edges # [2, num_edges]


def sparse_to_adj_np(sp_matrix):

    """
    Transfer sparse matrix to adjacency list
    :param sp_matrix: the sparse matrix
    :param filename: the filename of adjlist
    """
    # add self loop
    homo_adj = sp_matrix + sp.eye(sp_matrix.shape[0])
    # create adj_np_list
    edges = homo_adj.nonzero()
    src = torch.tensor(edges[0])
    dst = torch.tensor(edges[1])
    edges = torch.stack([src, dst], dim=0)
    edges = np.array(edges)
    return edges # [2, num_edges]


def mask_edge(e, mask_prob):
    E = e
    mask_rates = torch.FloatTensor(np.ones(E) * mask_prob)
    masks = torch.bernoulli(1 - mask_rates)
    mask_idx = masks.nonzero().squeeze(1)
    return mask_idx


def dgl_get_yelp_hetero(dataset, train_ratio):
    if dataset == 'YelpChi':
        yelp = scio.loadmat('YelpChi.mat')
    else:
        yelp = scio.loadmat('YelpNYC.mat')

    # edge
    net_rur = yelp['net_rur']  # format:csc
    net_rtr = yelp['net_rtr']
    net_rsr = yelp['net_rsr']
    rur_tensor = sparse_to_adj_np(net_rur)
    rtr_tensor = sparse_to_adj_np(net_rtr)
    rsr_tensor = sparse_to_adj_np(net_rsr)
    rur_tensor_src = rur_tensor[0, :]
    rur_tensor_dis = rur_tensor[1, :]
    rtr_tensor_src = rtr_tensor[0, :]
    rtr_tensor_dis = rtr_tensor[1, :]
    rsr_tensor_src = rsr_tensor[0, :]
    rsr_tensor_dis = rsr_tensor[1, :]
    # no-use
    dislike_src = np.random.randint(0, 1, 1)
    dislike_dst = np.random.randint(0, 1, 1)
    graph_data = {
        ('review', 'user', 'review'): (rur_tensor_src, rur_tensor_dis),
        ('review', 'time', 'review'): (rtr_tensor_src, rtr_tensor_dis),
        ('review', 'star', 'review'): (rsr_tensor_src, rsr_tensor_dis),
        ('user', 'dislike', 'item'): (dislike_src, dislike_dst)  # no-use
    }

    g = dgl.heterograph(graph_data)

    # node_feat
    if dataset == 'YelpChi':
        feat_data = yelp['features'].todense().A
    else:
        feat_data = yelp['features']
    feat_tensor = torch.from_numpy(feat_data)
    feat_tensor = torch.as_tensor(feat_tensor, dtype=torch.float)
    g.nodes['review'].data['feature'] = feat_tensor
    n_hetero_feat = feat_tensor.shape[1]

    # no-use
    g.nodes['item'].data['feature'] = torch.randn(1, n_hetero_feat)
    g.nodes['user'].data['feature'] = torch.randn(1, n_hetero_feat)

    # label
    labels = yelp['label'].flatten()
    label_tensor = torch.from_numpy(labels)
    g.nodes['review'].data['label'] = label_tensor
    n_nodes = label_tensor.shape[0]
    # true_index = np.argwhere(labels=True)

    # # mask
    # g.nodes['review'].data['train_mask'] = torch.rand(size=(n_nodes,)) < train_ratio
    #
    # # create balanced train node
    # train_true_tensor = torch.mul(label_tensor, g.nodes['review'].data['train_mask'])  # tensor([0, 0, 0,  ..., 0, 0, 0])
    #
    # train_false_tensor = g.nodes['review'].data['train_mask'] > label_tensor  # tensor([False, False, False,  ..., False,  True, False])
    #
    # train_true_index = np.argwhere(train_true_tensor)
    # train_true_index = train_true_index.squeeze(0)  # tensor([ 5984,  5985,  5992,  ..., 45864, 45870, 45879])
    # train_true_id = np.array(train_true_index)
    #
    # train_false_index = np.argwhere(train_false_tensor)
    # train_false_index = train_false_index.squeeze(0)
    # train_false_id = np.array(train_false_index)
    # random.shuffle(train_false_id)  # [11490 17633 22373 ... 30364 35657 19719]
    # random.shuffle(train_true_id)
    # # print('train_true_id:', train_true_id)
    # # print('shape: ', train_true_id.shape)
    # random.shuffle(train_true_id)
    # # print('after shuffle: ', train_true_id)
    # # print(train_true_id.shape)
    # # train_nid1 = np.append(train_false_id[0:int(train_false_id.shape[0]/6)], train_true_index)
    # # train_nid2 = np.append(train_false_id[int(train_false_id.shape[0]/6):2*int(train_false_id.shape[0]/6)], train_true_index)

    # g与g1节点数目相同，节点索引一致，特征、训练/测试集的分配一致
    # return g, n_hetero_feat, n_nodes, train_nid_dict, test_nid_dict
    return g, n_hetero_feat, n_nodes
    # return g, n_hetero_feat, n_nodes, train_nid_dict1, test_nid_dict

# dgl_get_yelp_hetero(0.2)


def dgl_get_yelp_hetero_aug(dataset, edge_mask_prob, feat_mask_prob):

    if dataset == 'YelpChi':
        yelp = scio.loadmat('YelpChi.mat')
    else:
        yelp = scio.loadmat('YelpNYC.mat')

    # edge
    net_rur = yelp['net_rur']  # format:csc
    net_rtr = yelp['net_rtr']
    net_rsr = yelp['net_rsr']
    rur_tensor = sparse_to_adj_np(net_rur)
    rtr_tensor = sparse_to_adj_np(net_rtr)
    rsr_tensor = sparse_to_adj_np(net_rsr)
    rur_tensor_src = rur_tensor[0, :]
    rur_tensor_dis = rur_tensor[1, :]
    rtr_tensor_src = rtr_tensor[0, :]
    rtr_tensor_dis = rtr_tensor[1, :]
    rsr_tensor_src = rsr_tensor[0, :]
    rsr_tensor_dis = rsr_tensor[1, :]
    # no-use
    dislike_src = np.random.randint(0, 1, 1)
    dislike_dst = np.random.randint(0, 1, 1)
    graph_data = {
        ('review', 'user', 'review'): (rur_tensor_src, rur_tensor_dis),
        ('review', 'time', 'review'): (rtr_tensor_src, rtr_tensor_dis),
        ('review', 'star', 'review'): (rsr_tensor_src, rsr_tensor_dis),
        ('user', 'dislike', 'item'): (dislike_src, dislike_dst)  # no-use
    }

    g = dgl.heterograph(graph_data)

    e1 = g.num_edges('user')
    e2 = g.num_edges('time')
    e3 = g.num_edges('star')
    # mask_prob = 0.2
    mask1 = mask_edge(e1, edge_mask_prob)
    mask2 = mask_edge(e2, edge_mask_prob)
    mask3 = mask_edge(e3, edge_mask_prob)

    (src1, dst1) = g.edges(etype='user')
    (src2, dst2) = g.edges(etype='time')
    (src3, dst3) = g.edges(etype='star')

    rur_src = src1[mask1]
    rur_dst = dst1[mask1]
    rtr_src = src2[mask2]
    rtr_dst = dst2[mask2]
    rsr_src = src3[mask3]
    rsr_dst = dst3[mask3]
    aug_graph_data = {
        ('review', 'user', 'review'): (rur_src, rur_dst),
        ('review', 'time', 'review'): (rtr_src, rtr_dst),
        ('review', 'star', 'review'): (rsr_src, rsr_dst),
        ('user', 'dislike', 'item'): (dislike_src, dislike_dst)  # no-use
    }
    g = dgl.heterograph(aug_graph_data)
    g = g.add_self_loop('user')
    g = g.add_self_loop('time')
    g = g.add_self_loop('star')

    # node_feat
    if dataset == 'YelpChi':
        feat_data = yelp['features'].todense().A
    else:
        feat_data = yelp['features']
    feat_tensor = torch.from_numpy(feat_data)
    feat_tensor = torch.as_tensor(feat_tensor, dtype=torch.float)
    g.nodes['review'].data['feature'] = feat_tensor

    n_hetero_feat = feat_tensor.shape[1]

    # drop feature
    # feat_mask_prob
    drop_mask = torch.empty((g.nodes['review'].data['feature'].size(1),),
                        dtype=torch.float32,).uniform_(0, 1) < feat_mask_prob
    g.nodes['review'].data['feature'] = g.nodes['review'].data['feature'].clone()
    g.nodes['review'].data['feature'][:, drop_mask] = 0

    # no-use
    g.nodes['item'].data['feature'] = torch.randn(1, n_hetero_feat)
    g.nodes['user'].data['feature'] = torch.randn(1, n_hetero_feat)

    # label
    # print(yelp['label'])
    # print(yelp['label'].shape)
    labels = yelp['label'].flatten()
    label_tensor = torch.from_numpy(labels)
    g.nodes['review'].data['label'] = label_tensor

    # g与g1节点数目相同，节点索引一致，特征、训练/测试集的分配一致
    return g


def dgl_get_amz_hetero(train_ratio):
    amz = scio.loadmat('D:/PycharmProjects/Fraud_Detection/baseline/Amazon.mat')

    # edge
    net_upu = amz['net_upu']
    # print('net_upu', net_upu)
    # print(type(amz))
    # print(type(net_upu))
    # print(net_upu.data)
    net_usu = amz['net_usu']
    net_uvu = amz['net_uvu']
    upu_tensor = sparse_to_adj_np(net_upu)
    usu_tensor = sparse_to_adj_np(net_usu)
    uvu_tensor = sparse_to_adj_np(net_uvu)
    upu_tensor_src = upu_tensor[0, :]
    # print(upu_tensor_src.shape[0])
    upu_tensor_dis = upu_tensor[1, :]
    usu_tensor_src = usu_tensor[0, :]
    usu_tensor_dis = usu_tensor[1, :]
    uvu_tensor_src = uvu_tensor[0, :]
    uvu_tensor_dis = uvu_tensor[1, :]
    # print(net_upu.data.shape[0])
    # print(upu_tensor_src.shape[0])
    # print(upu_tensor_dis)
    data = torch.ones(upu_tensor_src.shape[0])
    net_upu_coo = coo_array((data, (upu_tensor_src, upu_tensor_dis)), shape=(11944, 11944)).toarray()
    data = torch.ones(usu_tensor_src.shape[0])
    net_usu_coo = coo_array((data, (usu_tensor_src, usu_tensor_dis)), shape=(11944, 11944)).toarray()
    data = torch.ones(uvu_tensor_src.shape[0])
    net_uvu_coo = coo_array((data, (uvu_tensor_src, uvu_tensor_dis)), shape=(11944, 11944)).toarray()
    # matrix1 = (net_usu_coo > 0.5).astype(bool)
    # matrix2 = (net_upu_coo > 0.5).astype(bool)
    # matrix3 = (net_uvu_coo > 0.5).astype(bool)
    # matrix = matrix1 + matrix2 + matrix3
    # matrix = matrix2
    # print(matrix.shape)
    # net_upu_coo = net_upu_coo+net_upu_coo
    # print(net_upu_coo)
    # no-use
    dislike_src = np.random.randint(0, 1, 1)
    dislike_dst = np.random.randint(0, 1, 1)
    graph_data = {
        ('user', 'p', 'user'): (upu_tensor_src, upu_tensor_dis),
        ('user', 's', 'user'): (usu_tensor_src, usu_tensor_dis),
        ('user', 'v', 'user'): (uvu_tensor_src, uvu_tensor_dis),
        ('it', 'dislike', 'item'): (dislike_src, dislike_dst)  # no-use
    }

    g = dgl.heterograph(graph_data)

    # node_feat
    feat_data = amz['features'].todense().A
    feat_tensor = torch.from_numpy(feat_data)
    feat_tensor = torch.as_tensor(feat_tensor, dtype=torch.float)
    g.nodes['user'].data['feature'] = feat_tensor
    # print('feat: ', g.nodes['user'].data['feature'])
    # print(g.nodes['user'].data['feature'].shape)
    n_hetero_feat = feat_tensor.shape[1]

    # no-use
    g.nodes['item'].data['feature'] = torch.randn(1, n_hetero_feat)
    g.nodes['it'].data['feature'] = torch.randn(1, n_hetero_feat)

    # label
    labels = amz['label'].flatten()
    label_tensor = torch.from_numpy(labels)
    # label_tensor = label_tensor[3305:]
    g.nodes['user'].data['label'] = label_tensor
    # print(g.nodes['user'].data['label'])
    n_nodes = label_tensor.shape[0]
    # true_index = np.argwhere(labels=True)

    # mask
    g.nodes['user'].data['train_mask'] = torch.rand(size=(n_nodes,)) < train_ratio
    g.nodes['user'].data['train_mask'][:3306] = False
    # print('train_mask: ', g.nodes['user'].data['train_mask'])
    g.nodes['user'].data['test_mask'] = (1 - g.nodes['user'].data['train_mask'].int()).bool()
    g.nodes['user'].data['test_mask'][:3306] = False
    # print('test_mask: ', g.nodes['user'].data['test_mask'])
    # print(g.nodes['review'].data['train_mask'].sum())
    train_nid = np.array(np.where(g.nodes['user'].data['train_mask']==True)).ravel()
    # print('train_nid: ', train_nid)

    random.shuffle(train_nid)
    train_nid_dict = {'user': train_nid}
    test_nid_dict = {'user': np.array(np.where(g.nodes['user'].data['test_mask']==True)).ravel()}

    # g与g1节点数目相同，节点索引一致，特征、训练/测试集的分配一致
    # return g, n_hetero_feat, n_nodes, train_nid_dict, test_nid_dict
    return g, n_hetero_feat, n_nodes, test_nid_dict, train_nid_dict


def dgl_get_amz_hetero_aug(edge_mask_prob, feat_mask_prob):

    amz = scio.loadmat('Amazon.mat')

    # edge
    net_upu = amz['net_upu']
    net_usu = amz['net_usu']
    net_uvu = amz['net_uvu']
    upu_tensor = sparse_to_adj_np(net_upu)
    usu_tensor = sparse_to_adj_np(net_usu)
    uvu_tensor = sparse_to_adj_np(net_uvu)
    upu_tensor_src = upu_tensor[0, :]
    upu_tensor_dis = upu_tensor[1, :]
    usu_tensor_src = usu_tensor[0, :]
    usu_tensor_dis = usu_tensor[1, :]
    uvu_tensor_src = uvu_tensor[0, :]
    uvu_tensor_dis = uvu_tensor[1, :]
    # no-use
    dislike_src = np.random.randint(0, 1, 1)
    dislike_dst = np.random.randint(0, 1, 1)
    graph_data = {
        ('user', 'p', 'user'): (upu_tensor_src, upu_tensor_dis),
        ('user', 's', 'user'): (usu_tensor_src, usu_tensor_dis),
        ('user', 'v', 'user'): (uvu_tensor_src, uvu_tensor_dis),
        ('it', 'dislike', 'item'): (dislike_src, dislike_dst)  # no-use
    }

    g = dgl.heterograph(graph_data)

    e1 = g.num_edges('p')
    e2 = g.num_edges('s')
    e3 = g.num_edges('v')
    # mask_prob = 0.2
    mask1 = mask_edge(e1, edge_mask_prob)
    mask2 = mask_edge(e2, edge_mask_prob)
    mask3 = mask_edge(e3, edge_mask_prob)

    (src1, dst1) = g.edges(etype='p')
    (src2, dst2) = g.edges(etype='s')
    (src3, dst3) = g.edges(etype='v')

    upu_src = src1[mask1]
    upu_dst = dst1[mask1]
    usu_src = src2[mask2]
    usu_dst = dst2[mask2]
    uvu_src = src3[mask3]
    uvu_dst = dst3[mask3]
    aug_graph_data = {
        ('user', 'p', 'user'): (upu_src, upu_dst),
        ('user', 's', 'user'): (usu_src, usu_dst),
        ('user', 'v', 'user'): (uvu_src, uvu_dst),
        ('it', 'dislike', 'item'): (dislike_src, dislike_dst)  # no-use
    }
    g = dgl.heterograph(aug_graph_data, num_nodes_dict={'user': 11944, 'it': 1, 'item':1})
    g = g.add_self_loop('p')
    g = g.add_self_loop('s')
    g = g.add_self_loop('v')

    # node_feat
    feat_data = amz['features'].todense().A
    feat_tensor = torch.from_numpy(feat_data)
    feat_tensor = torch.as_tensor(feat_tensor, dtype=torch.float)
    g.nodes['user'].data['feature'] = feat_tensor

    n_hetero_feat = feat_tensor.shape[1]

    # drop feature
    # feat_mask_prob
    drop_mask = torch.empty((g.nodes['user'].data['feature'].size(1),),
                        dtype=torch.float32,).uniform_(0, 1) < feat_mask_prob
    g.nodes['user'].data['feature'] = g.nodes['user'].data['feature'].clone()
    g.nodes['user'].data['feature'][:, drop_mask] = 0

    # no-use
    g.nodes['item'].data['feature'] = torch.randn(1, n_hetero_feat)
    g.nodes['it'].data['feature'] = torch.randn(1, n_hetero_feat)

    # label
    labels = amz['label'].flatten()
    label_tensor = torch.from_numpy(labels)
    g.nodes['user'].data['label'] = label_tensor

    # g与g1节点数目相同，节点索引一致，特征、训练/测试集的分配一致
    return g