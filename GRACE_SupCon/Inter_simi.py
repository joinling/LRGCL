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
data_name = 'YelpNYC.mat' # 'Amazon.mat' or 'YelpChi.mat' 'YelpNYC.mat'
# 类间
mode = 'diff'  # if set to pos, it only compute two metrics for positive nodes
data = loadmat(data_name)

if data_name == 'YelpNYC.mat':
    # print(data['homo'])
    net_list = [data['net_rur'].nonzero(), data['net_rtr'].nonzero(),
                data['net_rsr'].nonzero(), data['homo'].nonzero()]
else:  # amazon dataset
    net_list = [data['net_upu'].nonzero(), data['net_usu'].nonzero(),
                data['net_uvu'].nonzero(), data['homo'].nonzero()]
feature = normalize(data['features'])
#feature = feature.todense().A

# feature = normalize(data['features']).toarray()
label = data['label'][0]

# extract the edges of positive nodes in each relation graph
pos_nodes = set(label.nonzero()[0].tolist())  # true节点的ID
neg_nodes = set(np.nonzero(label == 0)[0].tolist())  # false节点id

node_list = [set(net[0].tolist()) for net in net_list]  # 各关系下图上节点id

pos_node_list = [list(net_nodes.intersection(pos_nodes)) for net_nodes in node_list]  # 各关系下TRUE节点id
neg_node_list = [list(net_nodes.intersection(neg_nodes)) for net_nodes in node_list]
pos_idx_list = []
pos_idx_list1 = []
neg_idx_list = []
neg_idx_list1 = []

for net, pos_node in zip(net_list, pos_node_list):
    pos_idx_list.append(np.in1d(net[0], np.array(pos_node)).nonzero()[0])  # 某种关系下的图的TRUE节点索引
    # pos_idx_list1.append(np.in1d(net[1], np.array(pos_node)).nonzero()[0])  # 某种关系下的图的TRUE节点索引
    # pos_idx_list = pos_idx_list1 and pos_node_list

for net, neg_node in zip(net_list, neg_node_list):
    # neg_idx_list.append(np.in1d(net[0], np.array(neg_node)).nonzero()[0])  # 某种关系下的图的false节点索引
    neg_idx_list.append(np.in1d(net[1], np.array(neg_node)).nonzero()[0])  # 某种关系下的图的TRUE节点索引
    # neg_idx_list = neg_idx_list1 and neg_node_list  # 边两边都是neg_node的edge idx
# 测试一维数组的每个元素是否也存在于第二个数组中。np.in1d
mix_idx_list = []
for pos, neg in zip(pos_idx_list, neg_idx_list):
    mix_idx_list.append(set(pos) & set(neg))

feature_simi_list = []
label_simi_list = []
mix_edge_list = []
for net, mix_idx in zip(net_list, mix_idx_list):
    feature_simi = 0
    label_simi = 0
    if mode == 'diff':  # compute two metrics for positive nodes
        for idx in mix_idx:
            u, v = net[0][idx], net[1][idx]
            a = feature[u]

            feature_simi += np.exp(-1 * np.square(np.linalg.norm(feature[u] - feature[v])))
            label_simi += label[u] == label[v]

        feature_simi = feature_simi / len(mix_idx)
        # print('1')
        # print(len(mix_idx))
        # print(feature_simi)
        label_simi = label_simi / len(mix_idx)  # 类间应该是0
        mix_edge = len(mix_idx)/net[0].size

    else:  # compute two metrics for all nodes
        for u, v in zip(net[0].tolist(), net[1].tolist()):
            feature_simi += np.exp(-1 * np.square(np.linalg.norm(feature[u] - feature[v])))
            label_simi += label[u] == label[v]

        feature_simi = feature_simi / net[0].size
        label_simi = label_simi / net[0].size

    feature_simi_list.append(feature_simi)
    label_simi_list.append(label_simi)
    mix_edge_list.append(mix_edge)

print(f'feature_simi: {feature_simi_list}')
print(f'label_simi: {label_simi_list}')
print(f'mix_edge_per: {mix_edge_list}')

