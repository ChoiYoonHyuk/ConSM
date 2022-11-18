import numpy as np
import argparse
import random
import math
from tqdm import tqdm
from swd import swd

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid, WikipediaNetwork, Actor
from torch_geometric.utils import to_dense_adj
from typing import Union, Tuple, Optional
from torch_scatter import gather_csr, segment_csr

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='Dataset')
parser.add_argument('data', type=int, help='data selector')
args = parser.parse_args()
data_id = args.data

# Set true homophily ratio here
if data_id == 0:
    dataset = Planetoid(root='/tmp/Cora', name='Cora')
    conf_r = .8
elif data_id == 1:
    dataset = Planetoid(root='/tmp/Citeseer', name='Citeseer')
    conf_r = .8
elif data_id == 2:
    dataset = Planetoid(root='/tmp/Pubmed', name='Pubmed')
    conf_r = .8
elif data_id == 3:
    dataset = WikipediaNetwork(root='/tmp/Chameleon', name='chameleon')
    conf_r = .4
elif data_id == 4:
    dataset = WikipediaNetwork(root='/tmp/Squirrel', name='squirrel')
    conf_r = .4
elif data_id == 5:
    dataset = Actor(root='/tmp/Actor')
    conf_r = .5
    

data = dataset[0].to(device)

# Parameter save
save_idx = True
path = './' + dataset.root[5:] + '_' + str(conf_r) + '.pth'
print(dataset.root[5:])

num_class = dataset.num_classes

if dataset.root == '/tmp/Chameleon' or dataset.root == '/tmp/Squirrel' or dataset.root == '/tmp/Actor':
    num_class = 5
    labels = dict()
    
    for x in range(len(data.y)):
        label = int(data.y[x])
        
        try:
            labels[label].append(x)
        except KeyError:
            labels[label] = [x]

    train_mask, valid_mask, test_mask = [], [], []
    for c in range(5):
        train_mask.extend(labels[c][0:20])
        cut = int((len(labels[c]) - 20) / 2)
        valid_mask.extend(labels[c][20:20+cut])
        test_mask.extend(labels[c][20+cut:len(labels[c])])
    
    train, valid, test = [], [], []
    for x in range(len(data.y)):
        if x in train_mask:
            train.append(True)
            valid.append(False)
            test.append(False)
        elif x in valid_mask:
            train.append(False)
            valid.append(True)
            test.append(False)
        elif x in test_mask:
            train.append(False)
            valid.append(False)
            test.append(True)
        else:
            train.append(False)
            valid.append(False)
            test.append(True)
    
    data.train_mask, data.val_mask, data.test_mask = torch.tensor(train).to(device), torch.tensor(valid).to(device), torch.tensor(test).to(device)
else:
    data.val_mask = torch.logical_not(torch.logical_or(data.train_mask, data.test_mask))
  
# LL edges: 0, LU edges: 1, UU edges: 0.5
l_u_edges = torch.zeros(len(data.edge_index[0])).to(device)
for p in range(len(data.edge_index[0])):  
    e_1, e_2 = data.edge_index[0][p], data.edge_index[1][p]  
    
    if data.train_mask[e_1] == False and data.train_mask[e_2] == False:
        l_u_edges[p] = .5
    elif data.train_mask[e_1] == False or data.train_mask[e_2] == False:
        l_u_edges[p] = 1.0
        
# Show true homophily ratio of graph dataset
tmp = []
for p in range(len(data.edge_index[0])):
    l_1, l_2 = data.y[data.edge_index[0][p]], data.y[data.edge_index[1][p]]
    
    if l_1 == l_2:
        tmp.append(1.0)
    else:
        tmp.append(0.0)
edge_weights = torch.tensor(tmp).to(device)
homophily = float(sum(edge_weights) / len(data.edge_index[0]))
print('True Homophily : %.2f' % homophily)

# Set anchor nodes using training samples
fixed_anchor = dict()
for idx in range(len(data.train_mask)):
    if data.train_mask[idx]:
        l = int(data.y[idx])
        if l in fixed_anchor:
            fixed_anchor[l].append(idx)
        else:
            fixed_anchor[l] = [idx]

#################################################################
################### Used for fast aggregation ###################
def broadcast(src: torch.Tensor, other: torch.Tensor, dim: int):
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand(other.size())
    return src

def scatter_sum(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                out: Optional[torch.Tensor] = None,
                dim_size: Optional[int] = None) -> torch.Tensor:
    index = broadcast(index, src, dim)
    
    if out is None:
        size = list(src.size())
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
        
        return out.scatter_add_(dim, index, src)
    else:
        return out.scatter_add_(dim, index, src)

def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                 out: Optional[torch.Tensor] = None,
                 dim_size: Optional[int] = None) -> torch.Tensor:
    out = scatter_sum(src, index, dim, out, dim_size)
    dim_size = out.size(dim)

    index_dim = dim
    if index_dim < 0:
        index_dim = index_dim + src.dim()
    if index.dim() <= index_dim:
        index_dim = index.dim() - 1

    ones = torch.ones(index.size(), dtype=src.dtype, device=src.device)
    count = scatter_sum(ones, index, index_dim, None, dim_size)
    count[count < 1] = 1
    count = broadcast(count, out, dim)
    if out.is_floating_point():
        out.true_divide_(count)
    else:
        out.div_(count, rounding_mode='floor')
    return out
                    
def scatter(src: torch.Tensor, index: torch.Tensor, dim: int = -1, out: Optional[torch.Tensor] = None, dim_size: Optional[int] = None,
            reduce: str = "sum") -> torch.Tensor:
    if reduce == 'sum' or reduce == 'add':
        return scatter_sum(src, index, dim, out, dim_size)
    if reduce == 'mean':
        return scatter_mean(src, index, dim, out, dim_size)
#################################################################        

# Graph convolution layer with label propagation
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden = 64
        self.gcn_1 = GCNConv(dataset.num_node_features, self.hidden)
        self.gcn_2 = GCNConv(self.hidden, num_class)
        
    def forward(self, data, edge_weight):
        x, edge_index = data.x, data.edge_index
        
        x_out = F.dropout(F.relu(self.gcn_1(x, edge_index)))
        x = self.gcn_2(x_out, edge_index)
        
        # n_cut: certain proportion of entire edges
        n_cut = int(len(edge_weight) * conf_r)
        
        # Find threshold using edge_weight prediction
        thresh = torch.topk(edge_weight, n_cut)[0][n_cut-1]
        
        # Propagation
        cosine_sim = 1 - F.cosine_similarity(x[data.edge_index[0]], x[data.edge_index[1]])
        # Over threshold -> reduce dissimilarity, below threshold -> reduce similarity 
        l_prop = torch.where(edge_weight >= thresh, cosine_sim, 1 - cosine_sim)
        l_edge = torch.where(edge_weight >= thresh, edge_weight, 1 - edge_weight)
        # Label propagation loss
        l_loss = torch.mean(l_edge * l_prop * l_u_edges)
        # For Actor dataset, subgraph matching module is more important than GNN
        if data_id == 5:
            l_loss = torch.sum(l_edge * l_prop * l_u_edges)
        
        return F.log_softmax(x, dim=1), l_loss
                
# Subgraph matching layer
class Sub_Mat(torch.nn.Module):
    def __init__(self):
        super(Sub_Mat, self).__init__()
        self.hidden = 64
        self.enc = nn.Linear(dataset.num_node_features, self.hidden)
        self.pred = nn.Linear(self.hidden, num_class)
        
        self.dec = nn.Sequential(
            nn.Linear(self.hidden * num_class, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        
    def forward(self, data, ego, pos, neg, edge_weight, idx):
        x, edge_index = data.x, data.edge_index
        
        embed = F.dropout(F.relu(self.enc(x)))
        x_out = self.pred(embed)
        
        # Construct anchor nodes using training samples
        anchor = torch.zeros(num_class, self.hidden).to(device)
        for c in range(num_class):
            anchor[c] = torch.mean(embed[fixed_anchor[c]], 0)
        
        # Find disassortative edges
        n_cut = int(len(edge_weight) * conf_r)
        #edge_score = F.cosine_similarity(x_out[data.edge_index[0]], x_out[data.edge_index[1]])
        thresh = torch.topk(edge_weight, n_cut)[0][n_cut-1]
        prune_edges = torch.where(edge_weight >= thresh, 1, 0)
        
        # Reconstruct adjacency matrix
        adj_matrix = (to_dense_adj(data.edge_index, edge_attr=prune_edges) > 0).squeeze(0)
        adj_matrix.fill_diagonal_(True)
        # Use two hop neighbors
        hop_2 = torch.matmul(1.0 * adj_matrix, 1.0 * adj_matrix) + adj_matrix > 0
        hop_2.fill_diagonal_(True)
        
        # If set to training
        if idx == 0:
            # Sampled ego, positive, and negative
            pos_l, neg_l = torch.zeros(len(ego), 1).to(device), torch.zeros(len(ego), 1).to(device)
            
            for d in range(len(ego)):
                # Neighbor nodes
                e2, p2, n2 = embed[hop_2[ego[d]]], embed[hop_2[pos[d]]], embed[hop_2[neg[d]]]
                
                # Find closest anchor for egos
                cdist = torch.cdist(e2, anchor)
                index = cdist.min(dim=1)[1]
                e_reduct = scatter(e2, index, dim=-2, dim_size=num_class, reduce='mean').view(-1)
                
                # Find closest anchor for positive pairs
                cdist = torch.cdist(p2, anchor)
                index = cdist.min(dim=1)[1]
                p_reduct = scatter(p2, index, dim=-2, dim_size=num_class, reduce='mean').view(-1)
                
                # Find closest anchor for negative pairs
                cdist = torch.cdist(n2, anchor)
                index = cdist.min(dim=1)[1]
                n_reduct = scatter(n2, index, dim=-2, dim_size=num_class, reduce='mean').view(-1)
                
                out_p, out_n = (e_reduct - p_reduct) * (e_reduct - p_reduct), (e_reduct - n_reduct) * (e_reduct - n_reduct)
                
                # Similarity between positive and negative pairs
                pos_l[d] = self.dec(out_p)
                neg_l[d] = self.dec(out_n)
            
            # Should also return class probability of central nodes
            return pos_l, neg_l, F.log_softmax(x_out, dim=1)
        # If set to evaluation
        else:
            same_prob = torch.zeros(len(data.edge_index[0])).to(device)
            for idx in range(len(edge_index[0])):
                n1, n2 = edge_index[0][idx], edge_index[1][idx]
                
                node_1, node_2 = embed[hop_2[n1]], embed[hop_2[n2]]
                
                cdist = torch.cdist(node_1, anchor)
                index = cdist.min(dim=1)[1]
                n1_reduct = scatter(node_1, index, dim=-2, dim_size=num_class, reduce='mean').view(-1)
                
                cdist = torch.cdist(node_2, anchor)
                index = cdist.min(dim=1)[1]
                n2_reduct = scatter(node_2, index, dim=-2, dim_size=num_class, reduce='mean').view(-1)
                
                out = (n1_reduct - n2_reduct) * (n1_reduct - n2_reduct)
                pend = self.dec(out)
                same_prob[idx] = pend
            
            # Only return edge weight here 
            return same_prob     

lr = 1e-3
model = Net().to(device)
model.train()
optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

sub_mat = Sub_Mat().to(device)
sub_mat.train()
sub_optim = torch.optim.Adam(sub_mat.parameters(), lr=lr, weight_decay=5e-4)

loss_func = nn.MSELoss()
n_trains = sum(data.train_mask)
best_gnn, best_valid, best_acc = 0, 0, 0
# Initialize edge weight as 0 
edge_weight = torch.zeros(len(data.edge_index[0])).to(device)

########################################
# Construct sets for subgraph matching #
gam_samples = dict()
for idx in range(len(data.train_mask)):
    if data.train_mask[idx]:
        l = int(data.y[idx])
        if l in gam_samples:
            gam_samples[l].append(idx)
        else:
            gam_samples[l] = [idx]

samples = []
for c in range(num_class):
    samples.append(gam_samples[c])
########################################

# Training
for ee in tqdm(range(300)):
    n_trains = sum(data.train_mask)
    
    # Train subgraph matching module
    sub_mat.train()
    
    for idx in range(n_trains):                  
        l = data.y[idx]
        # Sample ego
        ego = [idx] * (len(samples[l]) - 1)
        indices = np.arange(int(num_class))
        
        # Sample positive pairs
        pos, tmp_neg = samples[l], [ele for i, ele in enumerate(samples) if i not in l]
        neg = []
        for a in tmp_neg:
            for b in a:
                neg.append(b)
        
        ego, pos, neg = torch.tensor(ego).to(device).view(-1), torch.tensor(pos).to(device).view(-1), torch.tensor(neg).to(device).view(-1)
        pos = pos[pos != idx]
        randint = torch.randint(0, len(pos), (len(samples[l]) - 1, ))
        pos = pos[randint]
        
        # Sample negative pairs
        randint = torch.randint(0, len(neg), (len(samples[l]) - 1, ))
        neg = neg[randint]
        pos, neg, out = sub_mat(data, ego, pos, neg, edge_weight, False)
        pos_l, neg_l = torch.empty((len(samples[l]) - 1, 1)).fill_(1.0).to(device), torch.empty((len(samples[l]) - 1, 1)).fill_(0.0).to(device)
        
        sub_optim.zero_grad() 
        loss = loss_func(pos, pos_l) + loss_func(neg, neg_l) + F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        sub_optim.step()        
    
    # Using trained module, retrieve edge weight
    sub_mat.eval()
    with torch.no_grad():
        edge_weight = sub_mat(data, 0, 0, 0, edge_weight, True)
    
    # Train GNN module
    model.train()            
    for idx in range(500):                  
        out, l_loss = model(data, edge_weight)
        
        optim.zero_grad()
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask]) + l_loss 
        loss.backward()
        optim.step()
        
        with torch.no_grad():
            model.eval()
            pred, _ = model(data, edge_weight)
            _, pred = pred.max(dim=1)
            correct = float(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
            acc = correct / data.test_mask.sum().item()
            valid = float(pred[data.val_mask].eq(data.y[data.val_mask]).sum().item())
            
            # Show best accuracy which is independent of validation score
            if acc > best_acc:
                best_acc = acc
            
            # Show accuracy with best validation score
            if valid > best_valid:
                best_valid = valid
                best_gnn = acc
                
                # If achieves best validation, save parameters
                if save_idx:
                    torch.save(model.state_dict(), path)
    
    # Load trained parameters
    if save_idx:
        model.load_state_dict(torch.load(path))
    
    # For general performance -> please use best_acc
    # If you employ validation score for evaluation (including all baselines) -> please use best_gnn
    print('Best acc: %.3f, Acc with best validation: %.3f' % (best_acc, best_gnn))
        