import torch
from torch import nn
import torch.nn.functional as F
# from graph_rewiring import KNN, add_edges, edge_sampling, GDCWrapper
from base_classes import BaseGNN
from model_configurations import set_block, set_function
from utils import DummyData, get_full_adjacency
from function_transformer_attention import SpGraphTransAttentionLayer
from torch_geometric.utils import softmax
import torch_sparse
from torch_geometric.utils.loop import add_remaining_self_loops
import numpy as np
from data import get_dataset
from utils import MaxNFEException, squareplus
from base_classes import ODEFunc
import wandb


class GrandDiscritizedBlock(ODEFunc):

    def __init__(self, in_features, out_features, opt, data, device):
        super(GrandDiscritizedBlock, self).__init__(opt, data, device)
        data = data.data
        self.trunc_alpha = opt['trunc_alpha']
        # self.k = opt['k']
        #     self.coeff = opt['trunc_coeff']

        if opt['self_loop_weight'] > 0:
            self.edge_index, self.edge_weight = add_remaining_self_loops(data.edge_index, data.edge_attr,
                                                                       fill_value=opt['self_loop_weight'])
        else:
            self.edge_index, self.edge_weight = data.edge_index, data.edge_attr
        self.edge_index = self.edge_index.to(device)
        try:
            self.edge_weight = self.edge_weight.to(device)
        except Exception as e:
            pass
        self.multihead_att_layer = SpGraphTransAttentionLayer(in_features, out_features, opt,
                                                              device, edge_weights=self.edge_weight).to(device)
    def multiply_attention(self, x, attention, v=None, transpose=False):
        # todo would be nice if this was more efficient
        if self.opt['mix_features']:
            vx = torch.mean(torch.stack(
                [torch_sparse.spmm(self.edge_index, attention[:, idx], v.shape[0], v.shape[0], v[:, :, idx]) for idx in
                 range(self.opt['heads'])], dim=0),
                 dim=0)
            ax = self.multihead_att_layer.Wout(vx)
        else:
            mean_attention = attention.mean(dim=1)
            edge_index = self.edge_index
            if transpose==True:
                edge_index, mean_attention = torch_sparse.transpose(self.edge_index, mean_attention, x.shape[0], x.shape[0])

#             print(x.size())

            #TESTING RIGHT-STOCHASTIC, OUTPUT SHOULD BE [1, 1, 1, ..., 1]
#             test = torch.ones(x.shape[0]).unsqueeze(-1).to('cuda')
#             test1 = torch_sparse.spmm(self.edge_index, mean_attention, x.shape[0], x.shape[0], test)
#             print(test1.min(), test1.max())
#             print("----")
            #END TEST
            
            ax = torch_sparse.spmm(edge_index, mean_attention, x.shape[0], x.shape[0], x)
        return ax

    def forward(self, x):  # t is needed when called by the integrator
#     if self.nfe > self.opt["max_nfe"]:
#       raise MaxNFEException

        self.nfe += 1
        attention, values = self.multihead_att_layer(x, self.edge_index)
        if self.opt['one_block']:
            return attention
        ax = self.multiply_attention(x, attention, values)
        
        if not self.opt['no_alpha_sigmoid']:
          alpha = torch.sigmoid(self.alpha_train)
        else:
          alpha = self.alpha_train
        # f = alpha * (ax - x)
        f = ax-x
        if self.opt['add_source']:
          f = f + self.beta_train * x

        trunc = torch.norm(x, dim=(-1), keepdim=True)
        trunc2 = torch.pow(trunc, self.trunc_alpha)
        #     trunc2[torch.abs(trunc) > self.coeff] = self.coeff
        f = f * trunc2
        return f

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'



class GrandDiscritizedNet(BaseGNN):
  def __init__(self, hidden_dim, opt, data, device):
    super(GrandDiscritizedNet, self).__init__(opt, data, device)
#    opt["add_source"] = True
    self.step_size = torch.Tensor([opt["step_size"]]).to(device)
    self.mol_list = nn.ModuleList()
    ###CREATE ONE BLOCK TO UNITE ALL
    if opt["one_block"]:
        grand_block = GrandDiscritizedBlock(opt["hidden_dim"], hidden_dim, opt, data, device).to(device) 
        self.mol_list.append(grand_block)
    else:
        self.mol_list.append(
            GrandDiscritizedBlock(opt["hidden_dim"], hidden_dim, opt, data, device).to(device)
        )
    
    self.opt = opt
    self.data = data.data
    self.device = device
    self.data_edge_index = data.data.edge_index.to(device)
    self.fa = get_full_adjacency(self.num_nodes).to(device)
    for _ in range(opt["depth"]-1):
        if not opt["one_block"]:
            self.mol_list.append(GrandDiscritizedBlock(opt["hidden_dim"], hidden_dim, opt, data, device).to(device))
    
    ###################################3333
#    self.data_edge_index = dataset.data.edge_index.to(device)
#    self.fa = get_full_adjacency(self.num_nodes).to(device)
    def forward(self, x, pos_encoding = None):
    # Encode
      if self.opt['use_labels']:
        y = x[:, -self.num_classes:]
        x = x[:, :-self.num_classes]

      out = x
      for i in range(len(self.mol_list)):
      #print(f"After layers number {i+1}")
        out = self.mol_list[i](out)
      return out 
class GrandExtendDiscritizedNet(GrandDiscritizedNet):
  def __init__(self, opt, data, device):
    super().__init__(opt["hidden_dim"], opt, data, device)
    self.discritize_type = opt["discritize_type"]
  def forward(self,x, pos_encoding=False, debug=False):
#    print(x.shape, " this is shape before doing anything")
    if self.opt['use_labels']:
      y = x[:, -self.num_classes:]
      x = x[:, :-self.num_classes]
   # print(x.shape)
    if self.opt['beltrami']:
      x = F.dropout(x, self.opt['input_dropout'], training=self.training)
      x = self.mx(x)
      p = F.dropout(pos_encoding, self.opt['input_dropout'], training=self.training)
      p = self.mp(p)
      x = torch.cat([x, p], dim=1)
    else:
      x = F.dropout(x, self.opt['input_dropout'], training=self.training)
#       print("after drop", x)
      x = self.m1(x)
#     print("After 2",x)
    if self.opt['use_mlp']:
      x = F.dropout(x, self.opt['dropout'], training=self.training)
      x = F.dropout(x + self.m11(F.relu(x)), self.opt['dropout'], training=self.training)
      x = F.dropout(x + self.m12(F.relu(x)), self.opt['dropout'], training=self.training)
    # todo investigate if some input non-linearity solves the problem with smooth deformations identified in the ANODE paper
#     print("after 3", x)

    if self.opt['use_labels']:
      x = torch.cat([x, y], dim=-1)

    if self.opt['batch_norm']:
      x = self.bn_in(x)

    # Solve the initial value problem of the ODE.
    if self.opt['augment']:
      c_aux = torch.zeros(x.shape).to(self.device)
      x = torch.cat([x, c_aux], dim=1)
    out = x
#     print(f"This is the output shape before forward those Blocks: {x.shape}")
    if debug==True:
        states = [out]
        diff_values = []
    attention = self.mol_list[0](out) 
    for i in range(self.opt['depth']):
      if self.discritize_type=="norm":
#         print(torch.norm(out, dim=(-1)).shape)
#         print(self.mol_list[i](out).shape)
#         print(out)
#         print(torch.norm(out, dim=(-1)))
#         print(torch.pow(torch.norm(out, dim=(-1)),1))
#         print("----")
#         # trunc_k1 = torch.pow(torch.norm(out, dim=(-1)).unsqueeze(1), trunc_alpha)
#         trunc_k1 = torch.pow(torch.norm(out, dim=(-1), keepdim=True), trunc_alpha)
#         trunc_k1[torch.abs(trunc_k1) > coeff] = coeff
#         k1 = self.mol_list[i](out) * trunc_k1
        
#         inp_k2 = out + self.step_size/2 * k1
#         # trunc_k2 = torch.pow(torch.norm(inp_k2, dim=(-1)).unsqueeze(1),trunc_alpha)
#         trunc_k2 = torch.pow(torch.norm(inp_k2, dim=(-1), keepdim=True), trunc_alpha)
#         trunc_k2[torch.abs(trunc_k2) > coeff] = coeff
#         k2 = self.mol_list[i](inp_k2) * trunc_k2
        
#         inp_k3 = out + self.step_size/2 * k2
#         # trunc_k3 = torch.pow(torch.norm(inp_k3, dim=(-1)).unsqueeze(1),trunc_alpha)
#         trunc_k3 = torch.pow(torch.norm(inp_k3, dim=(-1), keepdim=True), trunc_alpha)
#         trunc_k3[torch.abs(trunc_k3) > coeff] = coeff
#         k3 = self.mol_list[i](inp_k3) * trunc_k3
        
#         inp_k4 = out + self.step_size * k3
#         # trunc_k4 = torch.pow(torch.norm(inp_k4, dim=(-1)).unsqueeze(1),trunc_alpha)
#         trunc_k4 = torch.pow(torch.norm(inp_k4, dim=(-1), keepdim=True), trunc_alpha)
#         trunc_k4[torch.abs(trunc_k4) > coeff] = coeff
#         k4 = self.mol_list[i](inp_k4) * trunc_k4
        
#         out = out + self.step_size / 6 * (k1 + 2*k2 + 2*k3 + k4)
        if not self.opt['one_block']:
            out = out + self.step_size * self.mol_list[i](out)
        else:
            ax = self.mol_list[0].multiply_attention(out,attention)
            ax_t = self.mol_list[0].multiply_attention(out,attention, transpose=True)
            A_hat_x = ax - 1*out
            out = out + self.step_size * A_hat_x * (1/2)
        if debug==True:
            states.append(out)
            diff_values.append(self.mol_list[0](out))
        ####

      # elif self.discritize_type == "accumulate_norm":
      #   out = out + self.mol_list[i](out) * self.step_size * torch.norm(out, dim =(-1), keepdim=True) * torch.norm(x, dim = (-1), keepdim=True)
      # else:
      #   out = out + self.step_size * self.mol_list[i](out)
#      print(f"After layers number {i+1}")
    z = out
    if self.opt['augment']:
      z = torch.split(z, x.shape[1] // 2, dim=1)[0]

    # Activation.
    z = F.relu(z)

    if self.opt['fc_out']:
      z = self.fc(z)
      z = F.relu(z)

    # Dropout.
    z = F.dropout(z, self.opt['dropout'], training=self.training)

    # Decode each node embedding to get node label.
    z = self.m2(z)
    if debug==True:
        return z, states, (attention, self.mol_list[0].edge_index)
    return z

    #print("")
    #print(torch.norm(x, dim=(-1)))
    


######################################################33
if __name__ == "__main__":
  
  print(f"Test the grand_discritized file")
  opt = {'depth': 5,'use_cora_defaults': False, 'dataset': 'Cora', 'data_norm': 'rw', 'self_loop_weight': 1.0, 'use_labels': False, 'geom_gcn_splits': False, 'num_splits': 1, 'label_rate': 0.5, 'planetoid_split': False, 'hidden_dim': 16, 'fc_out': False, 'input_dropout': 0.5, 'dropout': 0.0, 'batch_norm': False, 'optimizer': 'adam', 'lr': 0.01, 'decay': 0.0005, 'epoch': 100, 'alpha': 1.0, 'alpha_dim': 'sc', 'no_alpha_sigmoid': False, 'beta_dim': 'sc', 'block': 'constant', 'function': 'laplacian', 'use_mlp': False, 'add_source': False, 'cgnn': False, 'time': 1.0, 'augment': False, 'method': None, 'step_size': 1, 'max_iters': 100, 'adjoint_method': 'adaptive_heun', 'adjoint': False, 'adjoint_step_size': 1, 'tol_scale': 1.0, 'tol_scale_adjoint': 1.0, 'ode_blocks': 1, 'max_nfe': 1000, 'no_early': False, 'earlystopxT': 3, 'max_test_steps': 100, 'leaky_relu_slope': 0.2, 'attention_dropout': 0.0, 'heads': 4, 'attention_norm_idx': 0, 'attention_dim': 64, 'mix_features': False, 'reweight_attention': False, 'attention_type': 'scaled_dot', 'square_plus': False, 'jacobian_norm2': None, 'total_deriv': None, 'kinetic_energy': None, 'directional_penalty': None, 'not_lcc': True, 'rewiring': None, 'gdc_method': 'ppr', 'gdc_sparsification': 'topk', 'gdc_k': 64, 'gdc_threshold': 0.0001, 'gdc_avg_degree': 64, 'ppr_alpha': 0.05, 'heat_time': 3.0, 'att_samp_pct': 1, 'use_flux': False, 'exact': False, 'M_nodes': 64, 'new_edges': 'random', 'sparsify': 'S_hat', 'threshold_type': 'topk_adj', 'rw_addD': 0.02, 'rw_rmvR': 0.02, 'rewire_KNN': False, 'rewire_KNN_T': 'T0', 'rewire_KNN_epoch': 5, 'rewire_KNN_k': 64, 'rewire_KNN_sym': False, 'KNN_online': False, 'KNN_online_reps': 4, 'KNN_space': 'pos_distance', 'beltrami': False, 'fa_layer': False, 'pos_enc_type': 'DW64', 'pos_enc_orientation': 'row', 'feat_hidden_dim': 64, 'pos_enc_hidden_dim': 32, 'edge_sampling': False, 'edge_sampling_T': 'T0', 'edge_sampling_epoch': 5, 'edge_sampling_add': 0.64, 'edge_sampling_add_type': 'importance', 'edge_sampling_rmv': 0.32, 'edge_sampling_sym': False, 'edge_sampling_online': False, 'edge_sampling_online_reps': 4, 'edge_sampling_space': 'attention', 'symmetric_attention': False, 'fa_layer_edge_sampling_rmv': 0.8, 'gpu': 0, 'pos_enc_csv': False, 'pos_dist_quantile': 0.001, 'discritize_type': 'norm', 'one_block':False, 'trunc_alpha':0, 'k':1}
  device = "cuda"
  dataset = get_dataset(opt, '../data', False)
  dataset.data = dataset.data.to(device, non_blocking=True)
#   print(type(dataset.data.x))
#   print(type(dataset.data))
  func = GrandExtendDiscritizedNet(opt, dataset, device).to(device)
  out = func(dataset.data.x)

