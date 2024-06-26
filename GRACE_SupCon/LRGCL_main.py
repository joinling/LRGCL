import torch
import torch.nn.functional as F
import numpy as np
import argparse
import time
import dgl
from sklearn.metrics import f1_score, recall_score, roc_auc_score, precision_score
from sklearn.model_selection import train_test_split
from RGAT import StochasticTwoLayerRGAT
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
                                                            train_size=args.train_ratio,
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
    # print(g.nodes[category].data['feature'])

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

            h1, h2 = [model.project(x) for x in [z1, z2]]

            z = model(g, {category: g.nodes[category].data['feature'], unrelevent1: g.nodes[unrelevent1].data['feature'],
                                   unrelevent2: g.nodes[unrelevent2].data['feature']}, category)
            # print('blocks1[-1].dstnodes', blocks1[-1].dstnodes[category].data[dgl.NID])
            # print('blocks1[-1].srcnodes', blocks1[-1].srcnodes[category].data[dgl.NID])

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
                rela_mask3 = torch.from_numpy((matrix < 1).astype(bool))
                rela_mask2 = torch.from_numpy((matrix > 0).astype(bool))*torch.from_numpy((matrix < 2).astype(bool))
                rela_mask1 = torch.from_numpy((matrix > 1).astype(bool))*torch.from_numpy((matrix < 3).astype(bool))
                rela_mask0 = torch.from_numpy((matrix > 2).astype(bool))
                mask0 = torch.ones(matrix.shape)
                # rela_mask = torch.stack((mask0.squeeze(), torch.from_numpy((matrix > 2).astype(bool)).squeeze()), dim=2 )
                rela_mask = torch.stack((rela_mask0.squeeze()
                                        , rela_mask1.squeeze()), dim=2)
                # rela_mask :[bsz, bsz, 4] 3种关系->2种->1种->0关系
                rela_mask = torch.cat((rela_mask, rela_mask2.unsqueeze(2)), dim=2)
                rela_mask = torch.cat((rela_mask, rela_mask3.unsqueeze(2)), dim=2)
                cumulative_loss = torch.tensor(0.0)
                max_loss_lower_layer = torch.tensor(float('-inf'))
                for l in range(0, rela_mask.shape[2]):
                    extra_pos_mask = torch.eq(blocks1[-1].dstdata['label'][category],
                                              blocks1[-1].dstdata['label'][category].unsqueeze(dim=1))
                    # print(rela_mask[:, :, l].shape) [bsz, bsz]
                    extra_pos_mask.fill_diagonal_(False)
                    extra_pos_mask = torch.cat([extra_pos_mask, extra_pos_mask], dim=1)
                    # print('layer_mask type', type(layer_mask))

                    extra_neg_mask = torch.ne(blocks1[-1].dstdata['label'][category],
                                              blocks1[-1].dstdata['label'][category].unsqueeze(dim=1))
                    layer_mask = extra_neg_mask * rela_mask[:, :, l]  # [bsz, bsz]

                    layer_mask.fill_diagonal_(False)
                    layer_mask = torch.cat([layer_mask, layer_mask], dim=1)
                    # Sup_Con loss
                    layer_loss = contrast_model(h1=h1, h2=h2, extra_pos_mask=extra_pos_mask, extra_neg_mask=layer_mask)
                    layer_loss = torch.max(max_loss_lower_layer, layer_loss)
                    cumulative_loss += torch.pow(2, torch.tensor(
                        1 / (l+1)).type(torch.float)) * layer_loss
                    max_loss_lower_layer = torch.max(
                        max_loss_lower_layer.to(layer_loss), layer_loss)
                cumulative_loss = cumulative_loss/rela_mask.shape[2]

            ################
            logits = model.MLP(z)
            # probs = logits.softmax(1)
            probs = F.log_softmax(logits, dim=1)
            CEloss = F.nll_loss(probs[train_mask], labels[train_mask].long())
            # CELoss + beta * Contrastive Loss
            total_loss = CEloss + args.beta * cumulative_loss
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
        auc, thres = get_best_auc(labels[val_mask], probs[val_mask])

        # print('thres: ', thres)
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
    parser = argparse.ArgumentParser(description='LRGCL')
    parser.add_argument("--dataset", type=str, default="yelp",
                        help="Dataset for this model (yelp/amazon/tfinance/tsocial)")
    parser.add_argument("--dtrain_ratio", type=float, default=0.2, help="Training ratio")
    parser.add_argument("--batch-size", type=int, default=1024,
                        help="Mini-batch size. If -1, use full graph training.Amazon-256，yelp-1024")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("--hid_dim", type=int, default=32, help="Hidden layer dimension")
    parser.add_argument("--homo", type=int, default=0, help="1 for Homo and 0 for Hetero")
    parser.add_argument("--model", type=str, default='RGCN', help="RGCN,RGAT...")
    parser.add_argument("--epoch", type=int, default=100, help="The max number of epochs")
    parser.add_argument("--linked_pos", type=bool, default=True, help='whether sample need to be linked')
    parser.add_argument("--linked_neg", type=bool, default=False, help='whether sample need to be linked')
    parser.add_argument('--edge_mask_prob',  type=float, default=0.2)
    parser.add_argument('--feat_mask_prob', type=float, default=0.2)
    parser.add_argument('--beta', type=float, default=0.001, help='the weight of Contrastive Loss')
    parser.add_argument('--temp', type=float, default=0.2, help='Temperature.')
    parser.add_argument("--run", type=int, default=1, help="Running times")

    args = parser.parse_args()
    print(args)
    dataset_name = args.dataset
    homo = args.homo
    h_feats = args.hid_dim

    if args.dataset == 'yelp':
        graph, in_feats, n_nodes, test_nid_dict, train_nid_dict = dgl_get_yelp_hetero(args.train_ratio)
        g1 = dgl_get_yelp_hetero_aug(args.edge_mask_prob, args.feat_mask_prob)
        g2 = dgl_get_yelp_hetero_aug(args.edge_mask_prob, args.feat_mask_prob)
        category = 'review'
        unrelevent1 = 'item'
        unrelevent2 = 'user'
    elif args.dataset == 'amazon':

        graph, in_feats, n_nodes, test_nid_dict, train_nid_dict = dgl_get_amz_hetero(args.train_ratio)
        g1 = dgl_get_amz_hetero_aug(args.edge_mask_prob, args.feat_mask_prob)
        g2 = dgl_get_amz_hetero_aug(args.edge_mask_prob, args.feat_mask_prob)
        category = 'user'
        unrelevent1 = 'it'
        unrelevent2 = 'item'

    num_classes = 2

    if args.run == 1:
        if homo:
            pass
        else:
            # etypes = {'net_upu', 'net_usu', 'net_uvu'}
            if args.model == 'RGAT':
                model = StochasticTwoLayerRGAT(in_feats, h_feats, out_feat=32, rel_names=graph.etypes)
            elif args.model == 'RGCN':
                model = StochasticTwoLayerRGCN(in_feats, h_feats, out_feat=64, rel_names=graph.etypes)
                # model = RGCN(in_feats, h_feats, num_classes, len(graph.canonical_etypes))

        auc, f1, rec = [0., 0., 0.], [0., 0., 0.], [0., 0., 0.]
        for i in range(0, 3, 1):
            print('Round {}'.format(i))
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
        print(
            'Test: AUC {:.4f}±{:.4f} MF1 {:.4f}±{:.4f} recall {:.4f}±{:.4f} '.format(mean_auc, std_auc, mean_f1, std_f1,
                                                                                     mean_rec, std_rec,))

