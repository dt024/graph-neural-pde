from torch_geometric.datasets import WebKB
import torch
import numpy as np
from torch_geometric.data import Data


DATA_PATH = '../data/'

def get_data(name, split=0):
  path = DATA_PATH+name
  dataset = WebKB(path,name=name)
  
  data = dataset[0]
  splits_file = np.load(f'{path}/{name}/raw/{name}_split_0.6_0.2_{split}.npz')
  train_mask = splits_file['train_mask']
  val_mask = splits_file['val_mask']
  test_mask = splits_file['test_mask']

  data.train_mask = torch.tensor(train_mask, dtype=torch.bool)
  data.val_mask = torch.tensor(val_mask, dtype=torch.bool)
  data.test_mask = torch.tensor(test_mask, dtype=torch.bool)

  return data

def get_new_hetero_data(name, split=0, device='cpu'):
    path = DATA_PATH+f'{name.replace("-", "_")}.npz'
    data = np.load(path)
    node_features = torch.tensor(data['node_features'])
    labels = torch.tensor(data['node_labels'])
    edges = torch.tensor(data['edges'].T)
    train_masks = torch.tensor(data['train_masks'][split])
    val_masks = torch.tensor(data['val_masks'][split])
    test_masks = torch.tensor(data['test_masks'][split])

    return Data(x=node_features, edge_index=edges, y=labels, train_mask=train_masks, val_mask=val_masks, test_mask=test_masks, num_classes=len(labels.unique()))
