#from utils import sparse_to_adjlist
import pickle
import scipy.sparse as sp
from collections import defaultdict
import scipy.io as scio
"""
	Read data and save the adjacency matrices to adjacency lists
	Paper: Reinforced Neighborhood Selection Guided Multi-Relational Graph Neural Networks
	Source: https://github.com/safe-graph/RioGNN
"""


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
		pickle.dump(adj_lists, file)
	file.close()

if __name__ == "__main__":

	prefix = 'D:\\PycharmProjects\\PyGCL\\data\\'

	# Yelp
	yelp = scio.loadmat('D:\\PycharmProjects\\PyGCL\\data\\YelpChi.mat')
	net_rur = yelp['net_rur']
	net_rtr = yelp['net_rtr']
	net_rsr = yelp['net_rsr']
	yelp_homo = yelp['homo']
	print(net_rur)
	# 3 Augment Views
	net_rur_rtr = net_rur +net_rtr
	net_rur_rsr = net_rur + net_rsr
	net_rsr_rtr = net_rsr + net_rtr
	#print(net_rur_rur)
	# net_rur_rur = net_rur+net_rur
	# a = net_rur_rur.max() =>2

	# A+A^2
	# net_rur2h = csc_matrix.dot(net_rur, net_rur)
	# net_rur2h = net_rur2h + net_rur
	# sparse_to_adjlist(net_rur2h, prefix + 'yelp_rur_adjlists.pickle')
	# relation
	sparse_to_adjlist(net_rur, prefix + 'yelp_rur_adjlists.pickle')
	sparse_to_adjlist(net_rtr, prefix + 'yelp_rtr_adjlists.pickle')
	sparse_to_adjlist(net_rsr, prefix + 'yelp_rsr_adjlists.pickle')
	sparse_to_adjlist(yelp_homo, prefix + 'yelp_homo_adjlists.pickle')
	# view
	sparse_to_adjlist(net_rur_rtr, prefix + 'yelp_rur_rtr_adjlists.pickle')
	sparse_to_adjlist(net_rur_rsr, prefix + 'yelp_rur_rsr_adjlists.pickle')
	sparse_to_adjlist(net_rsr_rtr, prefix + 'yelp_rsr_rtr_adjlists.pickle')
	# # Amazon
	# amz = scio.loadmat('D:\\PycharmProjects\\RioGNN\\data\\Amazon.mat')
	# net_upu = amz['net_upu']
	# net_usu = amz['net_usu']
	# net_uvu = amz['net_uvu']
	# amz_homo = amz['homo']
	#
	# sparse_to_adjlist(net_upu, prefix + 'amz_upu_adjlists.pickle')
	# sparse_to_adjlist(net_usu, prefix + 'amz_usu_adjlists.pickle')
	# sparse_to_adjlist(net_uvu, prefix + 'amz_uvu_adjlists.pickle')
	# sparse_to_adjlist(amz_homo, prefix + 'amz_homo_adjlists.pickle')

