from typing import *
import os
import torch
import dgl
import random
import numpy as np
import pickle
import random as rd
import numpy as np
import scipy.sparse as sp
from scipy.io import loadmat
import copy as cp
from collections import defaultdict
import scipy.io as scio
import os
import logging


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.enabled = False
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.benchmark = True #for accelerating the running


def split_dataset(dataset, split_mode, *args, **kwargs):
    assert split_mode in ['rand', 'ogb', 'wikics', 'preload']
    if split_mode == 'rand':
        assert 'train_ratio' in kwargs and 'test_ratio' in kwargs
        train_ratio = kwargs['train_ratio']
        test_ratio = kwargs['test_ratio']
        num_samples = dataset.x.size(0)
        train_size = int(num_samples * train_ratio)
        test_size = int(num_samples * test_ratio)
        indices = torch.randperm(num_samples)
        return {
            'train': indices[:train_size],
            'val': indices[train_size: test_size + train_size],
            'test': indices[test_size + train_size:]
        }
    elif split_mode == 'ogb':
        return dataset.get_idx_split()
    elif split_mode == 'wikics':
        assert 'split_idx' in kwargs
        split_idx = kwargs['split_idx']
        return {
            'train': dataset.train_mask[:, split_idx],
            'test': dataset.test_mask,
            'val': dataset.val_mask[:, split_idx]
        }
    elif split_mode == 'preload':
        assert 'preload_split' in kwargs
        assert kwargs['preload_split'] is not None
        train_mask, test_mask, val_mask = kwargs['preload_split']
        return {
            'train': train_mask,
            'test': test_mask,
            'val': val_mask
        }


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize(s):
    return (s.max() - s) / (s.max() - s.mean())


def build_dgl_graph(edge_index: torch.Tensor) -> dgl.DGLGraph:
    row, col = edge_index
    return dgl.graph((row, col))


def batchify_dict(dicts: List[dict], aggr_func=lambda x: x):
    res = dict()
    for d in dicts:
        for k, v in d.items():
            if k not in res:
                res[k] = [v]
            else:
                res[k].append(v)
    res = {k: aggr_func(v) for k, v in res.items()}
    return res


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
    #edges = np.array(edges)
    return edges

def sparse_to_adjlist(sp_matrix, filename):
    """
    Transfer sparse matrix to adjacency list
    :param sp_matrix: the sparse matrix
    :param filename: the filename of adjlist
    """
    # add self loop
    homo_adj = sp_matrix + sp.eye(sp_matrix.shape[0])
    # create adj_list
    adj_lists = defaultdict(set)
    edges = homo_adj.nonzero()
    for index, node in enumerate(edges[0]):
        adj_lists[node].add(edges[1][index])
        adj_lists[edges[1][index]].add(node)
    with open(filename, 'wb') as file:
        pickle.dump(edges, file)
    file.close()

"""
    handle data
    refer：https://github.com/safe-graph/RioGNN
"""

def load():
    prefix = 'D:\\PycharmProjects\\PyGCL\\data\\'

    # Yelp
    yelp = scio.loadmat('/data/YelpChi.mat')
    net_rur = yelp['net_rur']
    net_rtr = yelp['net_rtr']
    net_rsr = yelp['net_rsr']
    yelp_homo = yelp['homo']
    print(net_rur)
    # 3 Augment Views
    net_rur_rtr = net_rur + net_rtr
    net_rur_rsr = net_rur + net_rsr
    net_rsr_rtr = net_rsr + net_rtr
    return yelp_homo,net_rur,net_rtr,net_rsr,net_rur_rtr,net_rur_rsr,net_rsr_rtr

# yelp_homo,net_rur,net_rtr,net_rsr,net_rur_rtr,net_rur_rsr,net_rsr_rtr = load()
# yelp_homo = torch.from_numpy(net_rur.toarray())
# print(yelp_homo)
# print(type(yelp_homo ))
# print(yelp_homo.shape)
# 内存不足，分图load
def load_data_homo(data):
    """
    Load graph, feature, and label
    :param data: the dataset name.
    :returns: home and single-relation graphs, feature, label, index
    """

    prefix = '../data/'
    if data == 'yelp':
        data_file = loadmat(prefix + 'YelpChi.mat')
        labels = data_file['label'].flatten()
        feat_data = data_file['features'].todense().A
        print('Reading yelp_homo_adjlists.pickle')
        # load the preprocessed adj_lists
        with open(prefix + 'yelp_homo_adjlists.pickle', 'rb') as file:
            homo = pickle.load(file)
        # with open(prefix + 'yelp_rur_adjlists.pickle', 'rb') as file:
        #     relation1 = pickle.load(file)
        # with open(prefix + 'yelp_rtr_adjlists.pickle', 'rb') as file:
        #     relation2 = pickle.load(file)
        # with open(prefix + 'yelp_rsr_adjlists.pickle', 'rb') as file:
        #     relation3 = pickle.load(file)

        # with open(prefix + 'yelp_rur_rtr_adjlists.pickle', 'rb') as file:
        #     relation1n2 = pickle.load(file)
        # with open(prefix + 'yelp_rur_rsr_adjlists.pickle', 'rb') as file:
        #     relation1n3 = pickle.load(file)
        # with open(prefix + 'yelp_rsr_rtr_adjlists.pickle', 'rb') as file:
        #     relation2n3 = pickle.load(file)
        # # relations = [relation1, relation2, relation3]
        # view = [relation1n2, relation1n3, relation2n3]
        index = list(range(len(labels)))

        return homo , feat_data, labels, index

def load_data_rur_rtr(data):
        """
        Load graph, feature, and label
        :param data: the dataset name.
        :returns: home and single-relation graphs, feature, label, index
        """

        prefix = '../data/'
        if data == 'yelp':
            data_file = loadmat(prefix + 'YelpChi.mat')
            labels = data_file['label'].flatten()
            feat_data = data_file['features'].todense().A
            print(data_file)

            with open(prefix + 'yelp_rur_rtr_adjlists.pickle', 'rb') as file:
                view_rur_rtr = pickle.load(file)

            index = list(range(len(labels)))

            return view_rur_rtr, feat_data, labels, index


def load_data_rsr_rtr(data):
    """
    Load graph, feature, and label
    :param data: the dataset name.
    :returns: home and single-relation graphs, feature, label, index
    """

    prefix = '../data/'
    if data == 'yelp':
        data_file = loadmat(prefix + 'YelpChi.mat')
        labels = data_file['label'].flatten()
        feat_data = data_file['features'].todense().A
        print(data_file)

        with open(prefix + 'yelp_rsr_rtr_adjlists.pickle', 'rb') as file:
            view_rsr_rtr = pickle.load(file)

        index = list(range(len(labels)))

        return view_rsr_rtr, feat_data, labels, index

def load_data_rur_rsr(data):
        """
        Load graph, feature, and label
        :param data: the dataset name.
        :returns: home and single-relation graphs, feature, label, index
        """

        prefix = '../data/'
        if data == 'yelp':
            data_file = loadmat(prefix + 'YelpChi.mat')
            labels = data_file['label'].flatten()
            feat_data = data_file['features'].todense().A
            print(data_file)

            with open(prefix + 'yelp_rur_rsr_adjlists.pickle', 'rb') as file:
                view_rur_rsr = pickle.load(file)

            index = list(range(len(labels)))

            return view_rur_rsr, feat_data, labels, index
    # elif data == 'amazon':
    #     data_file = loadmat(prefix + 'Amazon.mat')
    #     labels = data_file['label'].flatten()
    #     feat_data = data_file['features'].todense().A
    #     # load the preprocessed adj_lists
    #     with open(prefix + 'amz_homo_adjlists.pickle', 'rb') as file:
    #         homo = pickle.load(file)
    #     with open(prefix + 'amz_upu_adjlists.pickle', 'rb') as file:
    #         relation1 = pickle.load(file)
    #     with open(prefix + 'amz_usu_adjlists.pickle', 'rb') as file:
    #         relation2 = pickle.load(file)
    #     with open(prefix + 'amz_uvu_adjlists.pickle', 'rb') as file:
    #         relation3 = pickle.load(file)
    #     relations = [relation1, relation2, relation3]
    #     # 0-3304 are unlabeled nodes
    #     index = list(range(len(labels)))
    #     #index = list(range(3305, len(labels)))
    #     #labels = labels[3305:]

import numpy as np
import scipy.sparse as sp
import torch

'''
先将所有由字符串表示的标签数组用set保存，set的重要特征就是元素没有重复，
因此表示成set后可以直接得到所有标签的总数，随后为每个标签分配一个编号，创建一个单位矩阵，
单位矩阵的每一行对应一个one-hot向量，也就是np.identity(len(classes))[i, :]，
再将每个数据对应的标签表示成的one-hot向量，类型为numpy数组
'''
def encode_onehot(labels):
    classes = set(labels)  # set() 函数创建一个无序不重复元素集
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in  # identity创建方矩阵
                    enumerate(classes)}     # 字典 key为label的值，value为矩阵的每一行
    # enumerate函数用于将一个可遍历的数据对象组合为一个索引序列
    labels_onehot = np.array(list(map(classes_dict.get, labels)),  # get函数得到字典key对应的value
                             dtype=np.int32)
    return labels_onehot
    # map() 会根据提供的函数对指定序列做映射
    # 第一个参数 function 以参数序列中的每一个元素调用 function 函数，返回包含每次 function 函数返回值的新列表
    #  map(lambda x: x ** 2, [1, 2, 3, 4, 5])
    #  output:[1, 4, 9, 16, 25]


def load_data2(path="./data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)  # 储存为csr型稀疏矩阵
    labels = encode_onehot(idx_features_labels[:, -1])
    #这里的label为onthot格式，如第一类代表[1,0,0,0,0,0,0]
    # content file的每一行的格式为 ： <paper_id> <word_attributes>+ <class_label>
    #    分别对应 0, 1:-1, -1
    # feature为第二列到倒数第二列，labels为最后一列

    # build graph
    # cites file的每一行格式为：  <cited paper ID>  <citing paper ID>
    # 根据前面的contents与这里的cites创建图，算出edges矩阵与adj 矩阵
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    # 由于文件中节点并非是按顺序排列的，因此建立一个编号为0-(node_size-1)的哈希表idx_map，
    # 哈希表中每一项为id: number，即节点id对应的编号为number
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    # edges_unordered为直接从边表文件中直接读取的结果，是一个(edge_num, 2)的数组，每一行表示一条边两个端点的idx
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),  # flatten：降维，返回一维数组
                     dtype=np.int32).reshape(edges_unordered.shape)
    # 边的edges_unordered中存储的是端点id，要将每一项的id换成编号。
    # 在idx_map中以idx作为键查找得到对应节点的编号，reshape成与edges_unordered形状一样的数组
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),  # coo型稀疏矩阵
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    # 根据coo矩阵性质，这一段的作用就是，网络有多少条边，邻接矩阵就有多少个1，
    # 所以先创建一个长度为edge_num的全1数组，每个1的填充位置就是一条边中两个端点的编号，
    # 即edges[:, 0], edges[:, 1]，矩阵的形状为(node_size, node_size)。


    # build symmetric adjacency matrix   论文里A^=(D~)^0.5 A~ (D~)^0.5这个公式
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # 对于无向图，邻接矩阵是对称的。上一步得到的adj是按有向图构建的，转换成无向图的邻接矩阵需要扩充成对称矩阵
    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))   # eye创建单位矩阵，第一个参数为行数，第二个为列数
    # 对应公式A~=A+IN

    # 分别构建训练集、验证集、测试集，并创建特征矩阵、标签向量和邻接矩阵的tensor，用来做模型的输入
    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    features = torch.FloatTensor(np.array(features.todense()))  # tensor为pytorch常用的数据结构
    labels = torch.LongTensor(np.where(labels)[1])
    #这里将onthot label转回index
    adj = sparse_mx_to_torch_sparse_tensor(adj)   # 邻接矩阵转为tensor处理

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))  # 对每一行求和
    r_inv = np.power(rowsum, -1).flatten()  # 求倒数
    r_inv[np.isinf(r_inv)] = 0.  # 如果某一行全为0，则r_inv算出来会等于无穷大，将这些行的r_inv置为0
    r_mat_inv = sp.diags(r_inv)  # 构建对角元素为r_inv的对角矩阵
    mx = r_mat_inv.dot(mx)
    # 用对角矩阵与原始矩阵的点积起到标准化的作用，原始矩阵中每一行元素都会与对应的r_inv相乘，最终相当于除以了sum
    return mx

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels) # 使用type_as(tesnor)将张量转换为给定类型的张量。
    correct = preds.eq(labels).double()  # 记录等于preds的label eq:equal
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):    # 把一个sparse matrix转为torch稀疏张量
    """
    numpy中的ndarray转化成pytorch中的tensor : torch.from_numpy()
    pytorch中的tensor转化成numpy中的ndarray : numpy()
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    # 不懂的可以去看看COO性稀疏矩阵的结构
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def set_logger(args, data_name):
    '''
    Write logs to checkpoint and console
    '''

    save_path = os.path.join(args.save_path, data_name)
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    log_file = os.path.join(save_path, '%s_%d.log' % (args.model_name,args.times))
    for i in range(10):
        if not os.path.isfile(log_file):
            break
        log_file = os.path.join(save_path, '%s_%d.log' % (args.model_name, args.times + i + 1))

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    logging.info('log save at : {}'.format(log_file))