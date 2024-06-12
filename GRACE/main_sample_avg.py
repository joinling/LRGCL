import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import argparse
import time
import dgl
from sklearn.metrics import f1_score, recall_score, roc_auc_score, precision_score
from sklearn.model_selection import train_test_split
# from RGAT import StochasticTwoLayerRGAT
from RGCN import StochasticTwoLayerRGCN
from dgl.dataloading import MultiLayerNeighborSampler, DataLoader
from GCL.models import DualBranchContrast
import GCL.losses as L
from yelp_aug_v5 import dgl_get_amz_hetero, dgl_get_yelp_hetero, dgl_get_yelp_hetero_aug, dgl_get_amz_hetero_aug
from sklearn.preprocessing import StandardScaler
from scipy.sparse import coo_array
import warnings


def train(model, g, g1, g2, args):
    if args.dataset == 'yelp':
        labels = g.nodes['review'].data['label']
        print(labels.sum())
        index = list(range(len(labels)))
        category = 'review'
        unrelevent1 = 'item'
        unrelevent2 = 'user'
        rela1 = 'star'
        rela2 = 'time'
        rela3 = 'user'

    elif args.dataset == 'amazon':
        labels = g.nodes['user'].data['label']
        index = list(range(3305, len(labels)))
        category = 'user'
        unrelevent1 = 'it'
        unrelevent2 = 'item'
        rela1 = 'p'
        rela2 = 's'
        rela3 = 'v'

    idx_train, idx_rest, y_train, y_rest = train_test_split(index, labels[index], stratify=labels[index],
                                                            train_size=args.dtrain_ratio,
                                                            shuffle=True)
    idx_valid, idx_test, y_valid, y_test = train_test_split(idx_rest, y_rest, stratify=y_rest,
                                                            test_size=0.67,
                                                            shuffle=True)
    train_mask = torch.zeros([len(labels)]).bool()
    val_mask = torch.zeros([len(labels)]).bool()
    test_mask = torch.zeros([len(labels)]).bool()

    train_mask[idx_train] = 1
    val_mask[idx_valid] = 1
    test_mask[idx_test] = 1
    print('train/dev/test samples: ', train_mask.sum().item(), val_mask.sum().item(), test_mask.sum().item())
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    best_auc, final_tf1, final_trec, final_tpre, final_tmf1, final_tauc = 0., 0., 0., 0., 0., 0.

    weight = (1 - labels[train_mask]).sum().item() / labels[train_mask].sum().item()
    print('cross entropy weight: ', weight)
    time_start = time.time()

    # Standard
    transfer = StandardScaler()
    a = transfer.fit_transform(g.nodes[category].data['feature'].numpy())
    g.nodes[category].data['feature'] = torch.from_numpy(a)

    smapler = MultiLayerNeighborSampler([6]*2)
    train_loader1 = DataLoader(g1, {category: idx_train}, smapler, batch_size=args.batch_size, drop_last=True, shuffle=False)
    train_loader2 = DataLoader(g2, {category: idx_train}, smapler, batch_size=args.batch_size, drop_last=True, shuffle=False)
    # val_loader = DataLoader(g, {category: idx_valid}, smapler, batch_size=args.batch_size, shuffle=False)
    contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=args.temp), mode='L2L', intraview_negs=True)

    for e in range(args.epoch):
        model.train()
        for (input_nodes1, output_nodes1, blocks1), (input_nodes2, output_node2s, blocks2) in zip(train_loader1, train_loader2):

            input_features1 = blocks1[0].srcdata['feature']
            input_features2 = blocks2[0].srcdata['feature']
            z1 = model(blocks1, input_features1, category)
            z2 = model(blocks2, input_features2, category)

            # h1, h2 = [model.project(x) for x in [z1, z2]]
            h1 = model.project(z1)
            h2 = model.project(z2)
            # h1 = z1
            # h2 = z2
            z = model(g, {category: g.nodes[category].data['feature'], unrelevent1: g.nodes[unrelevent1].data['feature'],
                                   unrelevent2: g.nodes[unrelevent2].data['feature']}, category)


            if args.linked_pos ==True:
                NID = blocks1[-1].dstnodes[category].data[dgl.NID]
                sg = dgl.node_subgraph(g, {category: NID})
                data = torch.ones(sg.edges(etype=rela1)[0].shape[0])  # 有边的coo为1
                rela1_coo = coo_array((data, sg.edges(etype=rela1)), shape=(args.batch_size, args.batch_size)).toarray()
                data = torch.ones(sg.edges(etype=rela2)[0].shape[0])
                rela2_coo = coo_array((data, sg.edges(etype=rela2)), shape=(args.batch_size, args.batch_size)).toarray()
                data = torch.ones(sg.edges(etype=rela3)[0].shape[0])
                rela3_coo = coo_array((data, sg.edges(etype=rela3)), shape=(args.batch_size, args.batch_size)).toarray()
                matrix1 = (rela1_coo > 0.5).astype(bool)+0
                matrix2 = (rela2_coo > 0.5).astype(bool)+0
                matrix3 = (rela3_coo > 0.5).astype(bool)+0

                matrix = matrix1 + matrix2 + matrix3
                rela_mask = torch.stack((torch.from_numpy(((matrix1 + matrix2 + matrix3) > 2).astype(bool)).unsqueeze(2)
                                         , torch.from_numpy(((matrix1 + matrix2) > 1).astype(bool)+((matrix1 + matrix3) > 1).astype(bool)+((matrix3 + matrix2) > 1).astype(bool))),
                                        dim=2)
                # rela_mask :(bsz, bsz, 3)
                rela_mask = torch.cat((rela_mask, torch.from_numpy(matrix.astype(bool))), dim=2)

                for l in range(0, rela_mask.shape[2]):
                    pass
                    pos_label_mask = torch.eq(blocks1[-1].dstdata['label'][category],
                                              blocks1[-1].dstdata['label'][category].unsqueeze(dim=1))
                    layer_mask = torch.eq(pos_label_mask, rela_mask[:, :, l].squeeze(dim=2))
                    print('layer_mask:', layer_mask)
                    print('layer_mask shape', layer_mask.shape)

                matrix = matrix+0  # tensor[0,...1]

                # g_sorted = dgl.transforms.sort_csr_by_tag(g, labels)
                # bias = [1.0, 0.0001]
                # sg = dgl.sampling.sample_neighbors_biased(g_sorted, NID, -1, bias, edge_dir='out')

                # extra mask
                # compute extra pos and neg masks for semi-supervised learning
                extra_pos_mask = torch.eq(blocks1[-1].dstdata['label'][category],
                                          blocks1[-1].dstdata['label'][category].unsqueeze(dim=1))
                # same label & 1-hot neighbor
                extra_pos_mask = extra_pos_mask + 0  # tensor[0,...1]
                extra_pos_mask = ((matrix + np.array(extra_pos_mask)) > 1).astype(bool)  # 且运算
                # print('extra_pos_mask: ', extra_pos_mask.sum())  # 4418 > 1024
                extra_pos_mask = torch.from_numpy(extra_pos_mask)

            else:
                # extra mask
                # compute extra pos and neg masks for semi-supervised learning
                extra_pos_mask = torch.eq(blocks1[-1].dstdata['label'][category],
                                          blocks1[-1].dstdata['label'][category].unsqueeze(dim=1))

            # construct extra supervision signals for only training samples
            # extra_pos_mask[~data.train_mask][:, ~data.train_mask] = False  <== no need！ all nodes are training smaples
            extra_pos_mask.fill_diagonal_(False)

            # pos_mask: [N, 2N] for both inter-view and intra-view samples
            extra_pos_mask = torch.cat([extra_pos_mask, extra_pos_mask], dim=1)
            # cat(..,dim=1) 按第二个维度拼接
            # fill inter-view positives only; pos_mask for intra-view samples should have zeros in diagonal
            extra_pos_mask.fill_diagonal_(True)

            if args.linked_neg ==True:
                NID = blocks1[-1].dstnodes[category].data[dgl.NID]
                sg = dgl.node_subgraph(g, {category: NID})
                data = torch.ones(sg.edges(etype=rela1)[0].shape[0])  # 有边的coo为1
                rela1_coo = coo_array((data, sg.edges(etype=rela1)), shape=(args.batch_size, args.batch_size)).toarray()
                data = torch.ones(sg.edges(etype=rela2)[0].shape[0])
                rela2_coo = coo_array((data, sg.edges(etype=rela2)), shape=(args.batch_size, args.batch_size)).toarray()
                data = torch.ones(sg.edges(etype=rela3)[0].shape[0])
                rela3_coo = coo_array((data, sg.edges(etype=rela3)), shape=(args.batch_size, args.batch_size)).toarray()
                matrix1 = (rela1_coo > 0.5).astype(bool)+0
                matrix2 = (rela2_coo > 0.5).astype(bool)+0
                matrix3 = (rela3_coo > 0.5).astype(bool)+0

                matrix = matrix1 + matrix2 + matrix3

                matrix = matrix+0  # tensor[0,...1]

                # g_sorted = dgl.transforms.sort_csr_by_tag(g, labels)
                # bias = [1.0, 0.0001]
                # sg = dgl.sampling.sample_neighbors_biased(g_sorted, NID, -1, bias, edge_dir='out')

                extra_neg_mask = torch.ne(blocks1[-1].dstdata['label'][category],
                                          blocks1[-1].dstdata['label'][category].unsqueeze(dim=1))
                # same label & 1-hot neighbor
                extra_neg_mask = extra_neg_mask + 0  # tensor[0,...1]
                extra_neg_mask = ((matrix + np.array(extra_neg_mask)) > 1).astype(bool)  # 且运算
                # print('extra_pos_mask: ', extra_pos_mask.sum())  # 4418 > 1024
                extra_neg_mask = torch.from_numpy(extra_neg_mask)

            else:
                # extra mask
                # compute extra pos and neg masks for semi-supervised learning
                extra_neg_mask = torch.ne(blocks1[-1].dstdata['label'][category],
                                          blocks1[-1].dstdata['label'][category].unsqueeze(dim=1))

            # .ne():not equal to

            # extra_neg_mask[~data.train_mask][:, ~data.train_mask] = True
            extra_neg_mask.fill_diagonal_(False)
            extra_neg_mask = torch.cat([extra_neg_mask, extra_neg_mask], dim=1)
            # extra_neg_mask = None

            # unSup_Con loss
            loss = contrast_model(h1=h1, h2=h2, extra_pos_mask=None, extra_neg_mask=None)



            ################
            logits = model.MLP(z)
            # probs = logits.softmax(1)
            probs = F.log_softmax(logits, dim=1)
            CEloss = F.nll_loss(probs[train_mask], labels[train_mask].long())
            # CELoss + beta * Contrastive Loss
            total_loss = CEloss + args.beta * loss
            ################

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        model.eval()
        logits = model.MLP(z)
        probs = logits.softmax(1)
        # print('probs: ', probs)
        # print(probs.shape)
        # print('logits', logits)
        # print(logits.shape)
        # # print(probs[val_mask])
        # print('labels', labels)
        # print(labels.shape)
        #f1, thres = get_best_f1(labels[val_mask], probs[val_mask])
        auc, thres = get_best_auc(labels[val_mask], probs[val_mask])

        # print('probs[train_mask]')
        # print(probs[train_mask].shape)
        preds = np.zeros_like(labels)
        preds[probs[:, 1] > thres] = 1
        # print('preds', preds)
        # print(preds.shape)
        trec = recall_score(labels[test_mask], preds[test_mask])
        tpre = precision_score(labels[test_mask], preds[test_mask])
        tmf1 = f1_score(labels[test_mask], preds[test_mask], average='macro')
        tauc = roc_auc_score(labels[test_mask], probs[test_mask][:, 1].detach().numpy())

        if best_auc < auc:
            best_auc = auc
            final_trec = trec
            final_tpre = tpre
            final_tmf1 = tmf1
            final_tauc = tauc
        print('Epoch {}, loss: {:.4f}, val AUC: {:.4f}, (best {:.4f})'.format(e, total_loss, auc, best_auc))

    time_end = time.time()
    print('time cost: ', time_end - time_start, 's')
    print('Test: recall {:.2f} PRE {:.2f} MF1 {:.2f} AUC {:.2f}'.format(final_trec * 100,
                                                                     final_tpre * 100, final_tmf1 * 100,
                                                                     final_tauc * 100))
    return final_tauc, final_tmf1, final_trec


def get_best_auc(labels, probs):
    best_auc, best_thre = 0, 0
    for thres in np.linspace(0.05, 0.95, 19):
        preds = np.zeros_like(labels)
        preds[probs[:, 1] > thres] = 1
        mauc = roc_auc_score(labels, preds)
        if mauc > best_auc:
            best_auc = mauc
            best_thre = thres
    return best_auc, best_thre


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description='GRACE_SupCon')
    parser.add_argument("--dataset", type=str, default="yelp",
                        help="Dataset for this model (yelp/amazon/tfinance/tsocial)")
    parser.add_argument("--ddataset", type=str, default="YelpNYC",
                        help="Dataset for this model (Chi/NYC)")
    parser.add_argument("--dtrain_ratio", type=float, default=0.2, help="Training ratio")
    parser.add_argument("--batch-size", type=int, default=2048,
                        help="Mini-batch size. If -1, use full graph training.Amazon-256，yelp-1024")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("--hid_dim", type=int, default=32, help="Hidden layer dimension")
    parser.add_argument("--homo", type=int, default=0, help="1 for Homo and 0 for Hetero")
    parser.add_argument("--model", type=str, default='RGCN', help="RGCN,RGAT...")
    parser.add_argument("--epoch", type=int, default=50, help="The max number of epochs")
    parser.add_argument("--linked_pos", type=bool, default=False, help='whether sample need to be linked')
    parser.add_argument("--linked_neg", type=bool, default=False, help='whether sample need to be linked')
    parser.add_argument('--edge_mask_prob',  type=float, default=0.2)
    parser.add_argument('--feat_mask_prob', type=float, default=0.2)
    parser.add_argument('--beta', type=float, default=0.5, help='the weight of Contrastive Loss, refer to rumor detection')
    parser.add_argument('--temp', type=float, default=0.2, help='Temperature.')
    parser.add_argument("--runtime", type=int, default=3, help="Running times")

    args = parser.parse_args()
    print(args)

    if args.runtime == 3:
        auc, f1, rec = [0., 0., 0.], [0., 0., 0.], [0., 0., 0.] # runtime3
    else:
        auc, f1, rec = [0., 0., 0., 0., 0.], [0., 0., 0., 0., 0.], [0., 0., 0., 0., 0.]
    for i in range(0, args.runtime, 1):
        print('Round {}'.format(i))
        dataset_name = args.dataset
        homo = args.homo
        h_feats = args.hid_dim

        if args.dataset == 'yelp':
            graph, in_feats, n_nodes = dgl_get_yelp_hetero(args.ddataset, args.dtrain_ratio)
            g1 = dgl_get_yelp_hetero_aug(args.ddataset, args.edge_mask_prob, args.feat_mask_prob)
            g2 = dgl_get_yelp_hetero_aug(args.ddataset, args.edge_mask_prob, args.feat_mask_prob)
            category = 'review'
            unrelevent1 = 'item'
            unrelevent2 = 'user'
        elif args.dataset == 'amazon':

            graph, in_feats, n_nodes, test_nid_dict, train_nid_dict = dgl_get_amz_hetero(args.dtrain_ratio)
            g1 = dgl_get_amz_hetero_aug(args.edge_mask_prob, args.feat_mask_prob)
            g2 = dgl_get_amz_hetero_aug(args.edge_mask_prob, args.feat_mask_prob)
            category = 'user'
            unrelevent1 = 'it'
            unrelevent2 = 'item'

        num_classes = 2

        if homo:
            pass
        else:
            # etypes = {'net_upu', 'net_usu', 'net_uvu'}
            if args.model == 'RGAT':
                pass
                # model = StochasticTwoLayerRGAT(in_feats, h_feats, out_feat=32, rel_names=graph.etypes)
            elif args.model == 'RGCN':
                model = StochasticTwoLayerRGCN(in_feats, h_feats, out_feat=64, rel_names=graph.etypes,
                                               feat_dim=in_feats)

        final_tauc, final_tmf1, final_trec = train(model, graph, g1, g2, args)
        auc[i] = final_tauc
        f1[i] = final_tmf1
        rec[i] = final_trec
    mean_auc = np.mean(auc)
    mean_f1 = np.mean(f1)
    mean_rec = np.mean(rec)
    std_auc = np.std(auc)
    std_f1 = np.std(f1)
    std_rec = np.std(rec)
    print('Test: AUC {:.4f}±{:.4f} MF1 {:.4f}±{:.4f} recall {:.4f}±{:.4f} '.format(mean_auc, std_auc, mean_f1, std_f1,
                                                                                mean_rec, std_rec,
                                                                                ))



