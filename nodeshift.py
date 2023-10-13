import torch
from torch_geometric.nn import GCNConv
from torch.nn import Linear
from torch.nn import BCEWithLogitsLoss
import networkx as nx
from torch_geometric.utils import from_networkx
from torch_geometric.transforms import RandomLinkSplit
import numpy as np
import random



class LinkPredictor(torch.nn.Module):
    def __init__(
        self,
        in_channels: int
    ):
        super(LinkPredictor, self).__init__()
        self.conv1 = GCNConv(in_channels, in_channels)
        self.conv2 = GCNConv(in_channels, in_channels)
        
        self.lin = Linear(in_channels, in_channels)

        self.loss_fn = BCEWithLogitsLoss()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.lin.reset_parameters()


    def forward(self, x, edge_index, edge_weight, edge_label_index):
        out = self.conv1(x, edge_index, edge_weight=edge_weight)
        out = out.relu()
        out = self.conv2(out, edge_index, edge_weight=edge_weight)
        out = out.relu()
        
        src = out[edge_label_index[0]]
        dst = out[edge_label_index[1]]
        had = torch.mul(src,dst)
        out = self.lin(had)
        out = torch.sum(out, dim=-1)
        
        return out

    def get_edge_embeddings(self, x, edge_index, edge_weight, edge_label_index):
        out = self.conv1(x, edge_index, edge_weight=edge_weight)
        out = out.relu()
        out = self.conv2(out, edge_index, edge_weight=edge_weight)
        out = out.relu()
        src = out[edge_label_index[0]]
        dst = out[edge_label_index[1]]
        had = torch.mul(src,dst)
        out = self.lin(had)
        return out

    def loss(self, pred, link_label):
        return self.loss_fn(pred, link_label)

from sklearn.metrics import *

def get_activity_vectors(node_embeddings):
    activities = []
    for t in range(len(node_embeddings)):
        nodes = node_embeddings[t]
        activity = torch.mean(nodes,dim=0)
        activities.append(activity)
    activities = [act.cpu().detach().numpy() for act in activities]
    activities_hat = [v / np.linalg.norm(v) for v in activities]
    return activities_hat 

def train_model(model, train_data, val_data, test_data, device,\
          optimizer, num_epochs=200):
    train_data = train_data.to(device)
    best_epoch = -1
    
    for epoch in range(num_epochs):
        model.train()
        ## Note
        ## 1. Zero grad the optimizer
        ## 2. Compute loss and backpropagate
        ## 3. Update the model parameters
        optimizer.zero_grad()
            

        pred = model(train_data.x, train_data.edge_index, train_data.edge_attr, train_data.edge_label_index)
        loss = model.loss(pred, train_data.edge_label.type_as(pred))

        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.

        ##########################################

        avgpr_score_train = test(model, train_data, device)
        avgpr_score_test = test(model, test_data, device)
        
    return model, avgpr_score_train, avgpr_score_test

def test(model, test_data, device):
    model.eval()

    test_data = test_data.to(device)

    h = model(test_data.x, test_data.edge_index, test_data.edge_attr, test_data.edge_label_index)
    
    pred_cont = torch.sigmoid(h).cpu().detach().numpy()

    label = test_data.edge_label.cpu().detach().numpy()
    
    avgpr_score = average_precision_score(label, pred_cont)
 
    return avgpr_score

def encode_over_time(dataset, device, folder, d=10, lr=0.01, weight_decay=5e-4):
    #seed = 12345
    #torch.manual_seed(seed)
    #torch.cuda.manual_seed_all(seed)
    #np.random.seed(12345)
    #random.seed(12345)
    #torch.backends.cudnn.benchmark = False
    #torch.backends.cudnn.deterministic = True
    #torch.cuda.empty_cache()
    avgprs = []
    for t in range(len(dataset)):
        model = LinkPredictor(in_channels=d).to(device)
        model.reset_parameters()
        optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay = weight_decay)
        snap = dataset[t]
        snap.to(device)
        split = RandomLinkSplit(num_val=0.15, num_test=0.15)
        train, val, test = split(snap)
        model, avgpr_train, avgpr_test = train_model(model, train, val, test, device, optimizer)
        #print(f'Timestamp: {t} AUPRC TRAIN: {avgpr_train}, AUPRC TEST: {avgpr_test}')
        if t%10==0:
            print(f'Timestamp {t} done')
        emb = model.get_edge_embeddings(snap.x, snap.edge_index, snap.edge_attr, snap.edge_index)
        #save node emb and ids
        torch.save(emb, f'{folder}/{t}_embeddings.pt')
        #torch.save(snap.ids, f'{folder}/{t}_ids.pt')
        avgprs.append(avgpr_test)
    return avgprs

from numpy import dot
from numpy.linalg import norm
def cosine_sim(a,b):
    return dot(a, b)/(norm(a)*norm(b))

def pyg_from_networkx_list(dataset, d=10):
    pyg_data = []
    for t in range(len(dataset)):
        nx.set_node_attributes(dataset[t], {node:node for node,node in zip(list(dataset[t].nodes), list(dataset[t].nodes))}, "ids")
        pyg_data_snap = from_networkx(dataset[t], group_edge_attrs=all)
        pyg_data_snap.x = torch.randn((pyg_data_snap.num_nodes, d))
        pyg_data_snap.edge_attr = pyg_data_snap.edge_attr.float()
        pyg_data.append(pyg_data_snap)
    return pyg_data