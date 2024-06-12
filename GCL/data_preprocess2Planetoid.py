from utils import *
import pickle as pkl
import pickle
import networkx as nx
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
import scipy.sparse as sp
import numpy as np
from collections import defaultdict


# homo
# prepare：adj, features, labels, idx_train, idx_val, idx_test
def load_data_from_raw(data, num_node, ratio_train, ratio_val, ratio_test):
    '''
    prepare for file :'x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph'
    每个pickle文件中的图都包含了所有节点，且顺序一致
    那么index可以作为节点的唯一标识？
    '''

    if data == 'yelp':
        homo, feat_data_homo, labels_homo, index_homo = load_data_homo('yelp')

    # feature
    features = sp.csr_matrix(feat_data_homo[:, 1:-1], dtype=np.float32)

    # labels
    labels_homo = labels_homo.reshape(45954, 1)
    labels = encode_onehot(labels_homo[:, -1])

    # idx
    idx = np.array(index_homo[:], dtype=np.int32)
    idx_map = idx

    # edges
    yelp = scio.loadmat('D:\\PycharmProjects\\PyGCL\\data\\YelpChi.mat')
    yelp_homo = yelp['homo']
    adj = yelp_homo.tocoo()  # csc_matrix -> coo_matrix,此时不是邻接矩阵
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)# build symmetric adjacency matrix

    # 原始制作过程没有进行feature normalize

    # num_node = 45954
    # ratio_train = 0.2
    # ratio_val = 0.2
    # ratio_test = 0.2
    # 分别构建训练集、验证集、测试集，并创建特征矩阵、标签向量和邻接矩阵的tensor，用来做模型的输入
    # 参考脚本中，train，test，val的并集没有用上所有节点
    idx_train = range(int(num_node * ratio_train))
    idx_val = range(int(num_node * ratio_train), int(num_node * (ratio_train + ratio_val)))
    idx_test = range(int(num_node * (ratio_train + ratio_val)), int(num_node * (ratio_train + ratio_val + ratio_test)))

    return adj, idx_train, idx_val, idx_test, features, labels


"""
Loads input data from gcn/data directory

ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
    (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
    object;
ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

All objects above must be saved using python pickle module.

:param dataset_str: Dataset name
:return: All data input files loaded (as well the training/test data).
"""


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


if __name__ == '__main__':
    adj, idx_train, idx_val, idx_test, features, labels = load_data_from_raw('yelp', 45954, 0.2, 0.2, 0.3)

    b = labels[:idx_test[0]]
    a = labels[idx_test[-1] + 1:]
    print(labels[idx_test[-1] + 1:])

    save_root = "../data"
    print('saving .x .y .tx .ty...')
    # save ...
    pickle.dump(features[idx_train], open(f"{save_root}/ind.yelp_homo.x", "wb"))
    pickle.dump(sp.vstack((features[:idx_test[0]], features[idx_test[-1] + 1:])),
                open(f"{save_root}/ind.yelp_homo.allx", "wb"))
    pickle.dump(features[idx_test], open(f"{save_root}/ind.yelp_homo.tx", "wb"))

    pickle.dump(labels[idx_train], open(f"{save_root}/ind.yelp_homo.y", "wb"))
    pickle.dump(labels[idx_test], open(f"{save_root}/ind.yelp_homo.ty", "wb"))
    pickle.dump(np.vstack((labels[:idx_test[0]], labels[idx_test[-1] + 1:])),
                open(f"{save_root}/ind.yelp_homo.ally", "wb"))

    with open(f'{save_root}/ind.yelp_homo.test.index', 'w') as f:
        for item in list(idx_test):
            f.write("%s\n" % item)

    # ori_graph
    array_adj = np.argwhere(adj.toarray())
    ori_graph = defaultdict(list)
    for edge in array_adj:
        ori_graph[edge[0]].append(edge[1])
    pickle.dump(ori_graph, open(f"{save_root}/ind.yelp_homo.graph", 'wb'))
    print('File preparation DONE !!!')

    # dataset_str = 'yelp_homo'
    # names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    # objects = []
    # for i in range(len(names)):
    #     with open("../data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
    #         if sys.version_info > (3, 0):
    #             objects.append(pkl.load(f, encoding='latin1'))
    #         else:
    #             objects.append(pkl.load(f))
    #
    # x, y, tx, ty, allx, ally, graph = tuple(objects)
    # test_idx_reorder = parse_index_file("../data/ind.{}.test.index".format(dataset_str))
    # test_idx_range = np.sort(test_idx_reorder)
    #
    # p_features = sp.vstack((allx[:test_idx_range[0]], tx, allx[test_idx_range[0]:])).tolil()
    # p_features[test_idx_reorder, :] = features[test_idx_range, :]
    #
    # o_adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    #
    # o_labels = np.vstack((ally[:test_idx_range[0]], ty, ally[test_idx_range[0]:]))
    # o_labels[test_idx_reorder, :] = labels[test_idx_range, :]
    #
    # idx_test = test_idx_range.tolist()
    # idx_train = range(len(y))
    # idx_val = range(len(y), len(y) + 500)  # ?????500
    #
    # train_mask = sample_mask(idx_train, labels.shape[0])
    # val_mask = sample_mask(idx_val, labels.shape[0])
    # test_mask = sample_mask(idx_test, labels.shape[0])
    #
    # y_train = np.zeros(labels.shape)
    # y_val = np.zeros(labels.shape)
    # y_test = np.zeros(labels.shape)
    # y_train[train_mask, :] = labels[train_mask, :]
    # y_val[val_mask, :] = labels[val_mask, :]
    # y_test[test_mask, :] = labels[test_mask, :]

    # view_rur_rtr, feat_data_ut, labels_ut, index_ut = load_data_rur_rtr('yelp')
    #
    # view_rsr_rtr, feat_data_st, labels_st, index_st = load_data_rsr_rtr('yelp')
