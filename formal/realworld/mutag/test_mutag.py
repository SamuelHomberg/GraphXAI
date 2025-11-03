import os, sys; sys.path.append('..')
import torch
from torch_geometric.datasets import TUDataset

from graphxai.explainers import IntegratedGradExplainer
#from graphxai.explainers.utils.visualizations import visualize_mol_explanation
from graphxai.gnn_models.graph_classification import train, test
from graphxai.gnn_models.graph_classification.gcn import GCN_2layer, GCN_3layer
from graphxai.gnn_models.graph_classification.gin import GIN_2layer, GIN_3layer
from graphxai.datasets import Benzene

from utils import get_dataset

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load data: ------------------------------------------
#dataset = Benzene(split_sizes = (0.8, 0.2, 0), seed = seed)
dataset = get_dataset('mutagenicity', device = device)
test_loader, _ = dataset.get_test_loader()
train_loader, _ = dataset.get_train_loader(batch_size=1)
val_loader, _ = dataset.get_val_loader()

# Train GNN model -------------------------------------
model = GIN_3layer(14, 32, 2).to(device)

mpath = os.path.join('model_weights', 'GIN_mutag_new_weights.pth')
model.load_state_dict(torch.load(mpath))

f1, precision, recall, auprc, auroc = test(model, train_loader)
print(f'Train F1: {f1:.4f}, Train AUROC: {auroc:.4f}')
f1, precision, recall, auprc, auroc = test(model, val_loader)
print(f'Val F1: {f1:.4f}, Val AUROC: {auroc:.4f}')
f1, precision, recall, auprc, auroc = test(model, test_loader)
f1, precision, recall, auprc, auroc = test(model, test_loader)
print(f'Test F1: {f1:.4f}, Test AUROC: {auroc:.4f}')