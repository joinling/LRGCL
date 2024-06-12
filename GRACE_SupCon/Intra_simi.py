from scipy.io import loadmat
import numpy as np
import scipy.sparse as sp


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1)) + 0.01
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


# load data
data_name = 'Amazon.mat' # 'Amazon.mat' or 'YelpChi.mat'
mode = 'same'  # if set to pos, it only compute two metrics for positive nodes
data = loadmat(data_name)

if data_name == 'YelpChi.mat':
    # print(data['homo'])
    net_list = [data['net_rur'].nonzero(), data['net_rtr'].nonzero(),
                data['net_rsr'].nonzero(), data['homo'].nonzero()]
else:  # amazon dataset
    net_list = [data['net_upu'].nonzero(), data['net_usu'].nonzero(),
                data['net_uvu'].nonzero(), data['homo'].nonzero()]
feature = normalize(data['features'])
feature = feature.todense().A
# feature = normalize(data['features']).toarray()
label = data['label'][0]

# extract the edges of positive nodes in each relation graph
pos_nodes = set(label.nonzero()[0].tolist())  # true节点的ID
neg_nodes = set(np.nonzero(label==0)[0].tolist())  # false节点id

node_list = [set(net[0].tolist()) for net in net_list]  # 各关系下图上节点id

pos_node_list = [list(net_nodes.intersection(pos_nodes)) for net_nodes in node_list]  # 各关系下TRUE节点id
neg_node_list = [list(net_nodes.intersection(neg_nodes)) for net_nodes in node_list]
pos_idx_list = []
pos_idx_list1 = []
neg_idx_list = []
neg_idx_list1 = []
all_pos_idx_list = []
all_neg_idx_list = []

for net, pos_node in zip(net_list, pos_node_list):
    pos_idx_list.append(np.in1d(net[0], np.array(pos_node)).nonzero()[0])  # 某种关系下的图的TRUE节点索引
    pos_idx_list1.append(np.in1d(net[1], np.array(pos_node)).nonzero()[0])  # 某种关系下的图的TRUE节点索引

for pos, pos1 in zip(pos_idx_list, pos_idx_list1):
    all_pos_idx_list.append(set(pos)&set(pos1))


for net, neg_node in zip(net_list, neg_node_list):
    neg_idx_list.append(np.in1d(net[0], np.array(neg_node)).nonzero()[0])  # 某种关系下的图的false节点索引
    neg_idx_list1.append(np.in1d(net[1], np.array(neg_node)).nonzero()[0])  # 某种关系下的图的TRUE节点索引
for neg, neg1 in zip(neg_idx_list, neg_idx_list1):
    all_neg_idx_list.append(set(neg)&set(neg1))

# 测试一维数组的每个元素是否也存在于第二个数组中。np.in1d
feature_simi_list = []
label_simi_list = []
print('compute two metrics for pos')
for net, pos_idx in zip(net_list, all_pos_idx_list):
    feature_simi = 0
    label_simi = 0
    if mode == 'same':  # compute two metrics for positive nodes
        for idx in pos_idx:
            u, v = net[0][idx], net[1][idx]
            feature_simi += np.exp(-1 * np.square(np.linalg.norm(feature[u] - feature[v])))
            label_simi += label[u] == label[v]

        feature_simi = feature_simi / len(pos_idx)
        label_simi = label_simi / len(pos_idx)
        # print(pos_idx.size)
        # print(type(pos_idx))

    else:  # compute two metrics for all nodes
        for u, v in zip(net[0].tolist(), net[1].tolist()):
            feature_simi += np.exp(-1 * np.square(np.linalg.norm(feature[u] - feature[v])))
            label_simi += label[u] == label[v]

        feature_simi = feature_simi / net[0].size
        label_simi = label_simi / net[0].size

    feature_simi_list.append(feature_simi)
    label_simi_list.append(label_simi)

print('compute two metrics for neg')
for net, neg_idx in zip(net_list, all_neg_idx_list):
    feature_simi = 0
    label_simi = 0
    if mode == 'same':  # compute two metrics for positive nodes
        for idx in neg_idx:
            u, v = net[0][idx], net[1][idx]
            feature_simi += np.exp(-1 * np.square(np.linalg.norm(feature[u] - feature[v])))
            label_simi += label[u] == label[v]

        feature_simi = feature_simi / len(neg_idx)
        label_simi = label_simi / len(neg_idx)

    else:  # compute two metrics for all nodes
        for u, v in zip(net[0].tolist(), net[1].tolist()):
            feature_simi += np.exp(-1 * np.square(np.linalg.norm(feature[u] - feature[v])))
            label_simi += label[u] == label[v]

        feature_simi = feature_simi / net[0].size
        label_simi = label_simi / net[0].size

    feature_simi_list.append(feature_simi)
    label_simi_list.append(label_simi)

print(f'feature_simi: {feature_simi_list}')
print(f'label_simi: {label_simi_list}')
