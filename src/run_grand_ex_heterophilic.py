import argparse
from distutils.log import debug
from statistics import mean
import gc
import numpy as np
import torch
from torch_geometric.nn import GCNConv, ChebConv  # noqa
import torch.nn.functional as F
import time
import tqdm
from data_heterophilic import get_data
from ogb.nodeproppred import Evaluator
from graph_rewiring import apply_KNN, apply_beltrami, apply_edge_sampling
from best_params import  best_params_dict
from grand_discritized import GrandExtendDiscritizedNet


import wandb



def get_optimizer(name, parameters, lr, weight_decay=0):
  if name == 'sgd':
    return torch.optim.SGD(parameters, lr=lr, weight_decay=weight_decay)
  elif name == 'rmsprop':
    return torch.optim.RMSprop(parameters, lr=lr, weight_decay=weight_decay)
  elif name == 'adagrad':
    return torch.optim.Adagrad(parameters, lr=lr, weight_decay=weight_decay)
  elif name == 'adam':
    return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
  elif name == 'adamax':
    return torch.optim.Adamax(parameters, lr=lr, weight_decay=weight_decay)
  else:
    raise Exception("Unsupported optimizer: {}".format(name))


def add_labels(feat, labels, idx, num_classes, device):
  onehot = torch.zeros([feat.shape[0], num_classes]).to(device)
  if idx.dtype == torch.bool:
    idx = torch.where(idx)[0]  # convert mask to linear index
  onehot[idx, labels.squeeze()[idx]] = 1

  return torch.cat([feat, onehot], dim=-1)


def get_label_masks(data, mask_rate=0.5):
  """
  when using labels as features need to split training nodes into training and prediction
  """
  if data.train_mask.dtype == torch.bool:
    idx = torch.where(data.train_mask)[0]
  else:
    idx = data.train_mask
  mask = torch.rand(idx.shape) < mask_rate
  train_label_idx = idx[mask]
  train_pred_idx = idx[~mask]
  return train_label_idx, train_pred_idx


def train(model, optimizer, data, pos_encoding=None):
  model.train()
  optimizer.zero_grad()
  feat = data.x
  if model.opt['use_labels']:
    train_label_idx, train_pred_idx = get_label_masks(data, model.opt['label_rate'])
    feat = add_labels(feat, data.y, train_label_idx, model.num_classes, model.device)
  else:
    train_pred_idx = data.train_mask
  train_label = data.y.squeeze()[data.train_mask]
  out, label_r, all_r = model(feat, pos_encoding=False, debug=False, isCalLoss=True, train_label=train_label, mask=data.train_mask)
  # out_model = model(feat, pos_encoding, debug = True)
  # out = out_model[0]
  # ##CALCULATING REGULAZATION FOR R
  # states = torch.stack(out_model[1],dim=0)
  # last_state = states[:,data.train_mask]
  # state_sin = torch.sin(last_state)
  # state_cos = torch.cos(last_state)
  # train_label = data.y.squeeze()[data.train_mask]
  # sin_label = [0 for _ in range(model.num_classes)]
  # cos_label = [0 for _ in range(model.num_classes)]
  # r_reg_loss = 0
  # label_r = all_r = 0
  # for m in range(1,2):
  #   sin_label_order = cos_label_order = 0
  #   for label in range(model.num_classes):
  #     if m==1:
  #       sin_label[label] = torch.mean(state_sin[:,train_label==label], dim=1)
  #       cos_label[label] = torch.mean(state_cos[:,train_label==label], dim=1)
  #       mean_feas = torch.mean(torch.ones_like(sin_label[label]) - torch.sqrt(cos_label[label]**2 + sin_label[label]**2), dim=-1)
  #       mean_time = torch.mean(mean_feas) 
  #       r_reg_loss +=  mean_time / model.num_classes
  #       label_r = torch.clone(r_reg_loss)
  #     avg_phi = torch.mean(last_state[:, train_label==label], dim=1)
  #     sin_label_order += torch.sin(m*avg_phi) 
  #     cos_label_order += torch.cos(m*avg_phi)  
  #   mean_feas = torch.mean((sin_label_order**2 + cos_label_order**2)**0.5 / model.num_classes , dim=-1)
  #   mean_time = torch.mean(mean_feas)/m
  #   r_reg_loss += mean_time 
  #   all_r += torch.clone(mean_time)

  # for label1 in range(model.num_classes):
  #   for label2 in range(label1+1, model.num_classes):
  #       r_reg_loss -= (torch.sum((sin_label[label1] - sin_label[label2])**2 + (cos_label[label1] - cos_label[label2])**2))
  
  if model.opt['dataset'] == 'ogbn-arxiv':
    lf = torch.nn.functional.nll_loss
    loss = lf(out.log_softmax(dim=-1)[data.train_mask], data.y.squeeze(1)[data.train_mask]) + label_r + all_r
  else:
    lf = torch.nn.CrossEntropyLoss()
    loss = lf(out[data.train_mask], data.y.squeeze()[data.train_mask]) + label_r + all_r
    # loss = label_r + all_r
  # if hasattr(model, 'odeblock') and model.odeblock.nreg > 0:  # add regularisation - slower for small data, but faster and better performance for large data
  #   reg_states = tuple(torch.mean(rs) for rs in model.reg_states)
  #   regularization_coeffs = model.regularization_coeffs

  #   reg_loss = sum(
  #     reg_state * coeff for reg_state, coeff in zip(reg_states, regularization_coeffs) if coeff != 0
  #   )
  #   loss = loss + reg_loss
  # try:
  #   model.fm.update(model.getNFE())
  #   model.resetNFE()
  #   model.bm.update(model.getNFE())
  #   model.resetNFE()
  # except Exception as e:
  #   pass
  loss.backward()
  torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
  optimizer.step()
  # return loss.item()
  torch.cuda.empty_cache()
  return loss.item(), label_r, all_r


def train_OGB(model, mp, optimizer, data, pos_encoding=None):
  model.train()
  optimizer.zero_grad()
  feat = data.x
  if model.opt['use_labels']:
    train_label_idx, train_pred_idx = get_label_masks(data, model.opt['label_rate'])

    feat = add_labels(feat, data.y, train_label_idx, model.num_classes, model.device)
  else:
    train_pred_idx = data.train_mask

  pos_encoding = mp(pos_encoding).to(model.device)
  out = model(feat, pos_encoding)

  if model.opt['dataset'] == 'ogbn-arxiv':
    lf = torch.nn.functional.nll_loss
    loss = lf(out.log_softmax(dim=-1)[data.train_mask], data.y.squeeze(1)[data.train_mask])
  else:
    lf = torch.nn.CrossEntropyLoss()
    loss = lf(out[data.train_mask], data.y.squeeze()[data.train_mask])
  if model.odeblock.nreg > 0:  # add regularisation - slower for small data, but faster and better performance for large data
    reg_states = tuple(torch.mean(rs) for rs in model.reg_states)
    regularization_coeffs = model.regularization_coeffs

    reg_loss = sum(
      reg_state * coeff for reg_state, coeff in zip(reg_states, regularization_coeffs) if coeff != 0
    )
    loss = loss + reg_loss

#  model.fm.update(model.getNFE())
#  model.resetNFE()
#  model.bm.update(model.getNFE())
#  model.resetNFE()
  loss.backward()
  optimizer.step()
  return loss.item()


@torch.no_grad()
def test(model, data, pos_encoding=None, opt=None, debug=False, isCalLoss=False):  # opt required for runtime polymorphism
  model.eval()
  feat = data.x
  if model.opt['use_labels']:
    feat = add_labels(feat, data.y, data.train_mask, model.num_classes, model.device)
  # logits, accs = model(feat, pos_encoding), []
  if isCalLoss:
    test_label = data.y.squeeze()[data.test_mask]
    out_model, label_r, all_r = model(feat, pos_encoding=False, debug=debug, isCalLoss=isCalLoss, train_label=test_label, mask=data.test_mask)
  else:
    out_model = model(feat, pos_encoding=False, debug=debug)

  logits, accs = out_model, []

  for _, mask in data('train_mask', 'val_mask', 'test_mask'):
    pred = logits[mask].max(1)[1]
    acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
    accs.append(acc)
  # outs = model(feat,debug=True)
  if debug==True:
    return accs, out_model, label_r, all_r
  if isCalLoss==True:
    return accs, label_r, all_r
  return accs

def print_model_params(model):
  print(model)
  for name, param in model.named_parameters():
    if param.requires_grad:
      print(name)
      print(param.data.shape)


@torch.no_grad()
def test_OGB(model, data, pos_encoding, opt):
  if opt['dataset'] == 'ogbn-arxiv':
    name = 'ogbn-arxiv'

  feat = data.x
  if model.opt['use_labels']:
    feat = add_labels(feat, data.y, data.train_mask, model.num_classes, model.device)

  evaluator = Evaluator(name=name)
  model.eval()

  out = model(feat, pos_encoding).log_softmax(dim=-1)
  y_pred = out.argmax(dim=-1, keepdim=True)

  train_acc = evaluator.eval({
    'y_true': data.y[data.train_mask],
    'y_pred': y_pred[data.train_mask],
  })['acc']
  valid_acc = evaluator.eval({
    'y_true': data.y[data.val_mask],
    'y_pred': y_pred[data.val_mask],
  })['acc']
  test_acc = evaluator.eval({
    'y_true': data.y[data.test_mask],
    'y_pred': y_pred[data.test_mask],
  })['acc']

  return train_acc, valid_acc, test_acc


def main(cmd_opt, split):
#   best_opt = best_params_dict[cmd_opt['dataset']]
#   opt = {**cmd_opt,**best_opt}
  opt = {**cmd_opt}
  wandb_name = f"step: {opt['step_size']} depth: {opt['depth']}  coupling_strength: {opt['coupling_strength']} hidden: {opt['hidden_dim']}"
  num_run = f"split: {split}"
  if opt["one_block"]:
     group_name = 'Kuramoto_' + opt['dataset'] + 'OneBlock'
  else:
     group_name = 'RK4' + opt['dataset'] + 'final'
 
  print(wandb_name, group_name, num_run)
  wandb.init(project="my_grand", entity="ductuan024", name=num_run, group=group_name, job_type=wandb_name, reinit=True)
  wandb.config = opt
  print(opt['step_size'])

  dataset = get_data(opt['dataset'], split)
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  pos_encoding = None
#
#  if opt['rewire_KNN'] or opt['fa_layer']:
#    model = GNN_KNN(opt, dataset, device).to(device) if opt["no_early"] else GNNKNNEarly(opt, dataset, device).to(device)
#  else:
#    model = GNN(opt, dataset, device).to(device) if opt["no_early"] else GNNEarly(opt, dataset, device).to(device)
  model = GrandExtendDiscritizedNet(opt, dataset, device).to(device)

  data = dataset.to(device)

  parameters = [p for p in model.parameters() if p.requires_grad]
  print_model_params(model)
  optimizer = get_optimizer(opt['optimizer'], parameters, lr=opt['lr'], weight_decay=opt['decay'])
  best_time = best_epoch = train_acc = val_acc = test_acc = 0

  this_test = test_OGB if opt['dataset'] == 'ogbn-arxiv' else test
  for epoch in tqdm.tqdm(range(1, opt['epoch'] + 1)):
    start_time = time.time()

    if opt['rewire_KNN'] and epoch % opt['rewire_KNN_epoch'] == 0 and epoch != 0:
      ei = apply_KNN(data, pos_encoding, model, opt)
      model.odeblock.odefunc.edge_index = ei
    loss, label_r, between_r = train(model, optimizer, data, pos_encoding)
    tmp_train_acc, tmp_val_acc, tmp_test_acc = this_test(model, data, pos_encoding, opt)

    best_time = opt['time']
    if tmp_val_acc > val_acc:
      best_epoch = epoch
      train_acc = tmp_train_acc
      val_acc = tmp_val_acc
      test_acc = tmp_test_acc
      best_time = opt['time']
      best_label_r = label_r
      best_between_r = between_r
#    if not opt['no_early'] and model.odeblock.test_integrator.solver.best_val > val_acc:
#      best_epoch = epoch
#      val_acc = model.odeblock.test_integrator.solver.best_val
#      test_acc = model.odeblock.test_integrator.solver.best_test
#      train_acc = model.odeblock.test_integrator.solver.best_train
#      best_time = model.odeblock.test_integrator.solver.best_time

    log = 'Epoch: {:03d}, Runtime {:03f}, Loss {:03f}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}, InLabel_R: {:.4f}, Between_R: {:.4f}, Best time: {:.4f}'

    print(log.format(epoch, time.time() - start_time, loss, tmp_train_acc, tmp_val_acc, tmp_test_acc, label_r, between_r, best_time))
    wandb.log(
        {
            'train_acc': tmp_train_acc,
            'test_acc': tmp_test_acc,
            'val_acc': tmp_val_acc,
            'label_r':label_r,
            'between_r':between_r,
            'loss': loss
        }
    )
    gc.collect()
    torch.cuda.empty_cache()
  print('best val accuracy {:03f} with test accuracy {:03f} at epoch {:d} and best inlabel_r {:03f}, between_r {:03f}'.format(val_acc, 
                                                                                                     test_acc,
                                                                                                     best_epoch,
                                                                                                     best_label_r,
                                                                                                     best_between_r))
  wandb.log(
      {
          'best_epoch' : best_epoch,
          'best_val' : val_acc,
          'best_test' : test_acc,
          'best_label_r': best_label_r,
          'best_between_r': best_between_r,
      }
           )
  return train_acc, val_acc, test_acc


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--use_cora_defaults', action='store_true',
                      help='Whether to run with best params for cora. Overrides the choice of dataset')
  # data args
  parser.add_argument('--dataset', type=str, default='texas',
                      help='texas, cornell, wisconsin')
  parser.add_argument('--data_norm', type=str, default='rw',
                      help='rw for random walk, gcn for symmetric gcn norm')
  parser.add_argument('--self_loop_weight', type=float, default=1.0, help='Weight of self-loops.')
  parser.add_argument('--use_labels', dest='use_labels', action='store_true', help='Also diffuse labels')
  parser.add_argument('--label_rate', type=float, default=0.5,
                      help='% of training labels to use when --use_labels is set.')
  # GNN args
  parser.add_argument('--hidden_dim', type=int, default=32, help='Hidden dimension.')
  parser.add_argument('--fc_out', dest='fc_out', action='store_true',
                      help='Add a fully connected layer to the decoder.')
  parser.add_argument('--input_dropout', type=float, default=0.2, help='Input dropout rate.')
  parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate.')
  parser.add_argument("--batch_norm", dest='batch_norm', action='store_true', help='search over reg params')
  parser.add_argument('--optimizer', type=str, default='adamax', help='One from sgd, rmsprop, adam, adagrad, adamax.')
  parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate.')
  parser.add_argument('--decay', type=float, default=1e-3, help='Weight decay for optimization')
  parser.add_argument('--epoch', type=int, default=500, help='Number of training epochs per iteration.')
  parser.add_argument('--alpha', type=float, default=0.8, help='Factor in front matrix A.')
  parser.add_argument('--alpha_dim', type=str, default='sc', help='choose either scalar (sc) or vector (vc) alpha')


  ### discritized param
  parser.add_argument('--depth', type=int, default=128, help='Default depth of the network')
  parser.add_argument('--discritize_type', type=str, default="norm", help="norm or acc_norm")
  parser.add_argument('--one_block', action='store_true', help='perform Linear Attention')
  parser.add_argument('--coupling_strength', type=float, default=3.0, help='Kuramoto coupling strength')

  ################# end of discritized param
  parser.add_argument('--no_alpha_sigmoid', dest='no_alpha_sigmoid', action='store_true',
                      help='apply sigmoid before multiplying by alpha')
  parser.add_argument('--beta_dim', type=str, default='sc', help='choose either scalar (sc) or vector (vc) beta')
  parser.add_argument('--block', type=str, default='constant', help='constant, mixed, attention, hard_attention')
  parser.add_argument('--function', type=str, default='laplacian', help='laplacian, transformer, dorsey, GAT')
  parser.add_argument('--use_mlp', dest='use_mlp', action='store_true',
                      help='Add a fully connected layer to the encoder.')
  parser.add_argument('--add_source', dest='add_source', action='store_true',
                      help='If try get rid of alpha param and the beta*x0 source term')

  # ODE args
  parser.add_argument('--time', type=float, default=100.0, help='End time of ODE integrator.')
  parser.add_argument('--augment', action='store_true',
                      help='double the length of the feature vector by appending zeros to stabilist ODE learning')
  parser.add_argument('--method', type=str, default='dopri5',
                      help="set the numerical solver: dopri5, euler, rk4, midpoint")
  parser.add_argument('--step_size', type=float, default=0.01,
                      help='fixed step size when using fixed step solvers e.g. rk4')
  parser.add_argument('--max_iters', type=float, default=100, help='maximum number of integration steps')
  parser.add_argument("--adjoint_method", type=str, default="adaptive_heun",
                      help="set the numerical solver for the backward pass: dopri5, euler, rk4, midpoint")
  parser.add_argument('--adjoint', dest='adjoint', action='store_true',
                      help='use the adjoint ODE method to reduce memory footprint')
  parser.add_argument('--adjoint_step_size', type=float, default=1,
                      help='fixed step size when using fixed step adjoint solvers e.g. rk4')
  parser.add_argument('--tol_scale', type=float, default=1., help='multiplier for atol and rtol')
  parser.add_argument("--tol_scale_adjoint", type=float, default=1.0,
                      help="multiplier for adjoint_atol and adjoint_rtol")
  parser.add_argument('--ode_blocks', type=int, default=1, help='number of ode blocks to run')
  parser.add_argument("--max_nfe", type=int, default=10000,
                      help="Maximum number of function evaluations in an epoch. Stiff ODEs will hang if not set.")
  parser.add_argument("--no_early", action="store_true",
                      help="Whether or not to use early stopping of the ODE integrator when testing.")
  parser.add_argument('--earlystopxT', type=float, default=3, help='multiplier for T used to evaluate best model')
  parser.add_argument("--max_test_steps", type=int, default=100,
                      help="Maximum number steps for the dopri5Early test integrator. "
                           "used if getting OOM errors at test time")

  # Attention args
  parser.add_argument('--leaky_relu_slope', type=float, default=0.15,
                      help='slope of the negative part of the leaky relu used in attention')
  parser.add_argument('--attention_dropout', type=float, default=0.2, help='dropout of attention weights')
  parser.add_argument('--heads', type=int, default=4, help='number of attention heads')
  parser.add_argument('--attention_norm_idx', type=int, default=0, help='0 = normalise rows, 1 = normalise cols')
  parser.add_argument('--attention_dim', type=int, default=64,
                      help='the size to project x to before calculating att scores')
  parser.add_argument('--mix_features', dest='mix_features', action='store_true',
                      help='apply a feature transformation xW to the ODE')
  parser.add_argument('--reweight_attention', dest='reweight_attention', action='store_true',
                      help="multiply attention scores by edge weights before softmax")
  parser.add_argument('--attention_type', type=str, default="scaled_dot",
                      help="scaled_dot,cosine_sim,pearson, exp_kernel")
  parser.add_argument('--square_plus', action='store_true', help='replace softmax with square plus')

  # regularisation args
  parser.add_argument('--jacobian_norm2', type=float, default=None, help="int_t ||df/dx||_F^2")
  parser.add_argument('--total_deriv', type=float, default=None, help="int_t ||df/dt||^2")

  parser.add_argument('--kinetic_energy', type=float, default=None, help="int_t ||f||_2^2")
  parser.add_argument('--directional_penalty', type=float, default=None, help="int_t ||(df/dx)^T f||^2")

  # rewiring args
  parser.add_argument("--not_lcc", action="store_false", help="don't use the largest connected component")
  parser.add_argument('--rewiring', type=str, default=None, help="two_hop, gdc")
  parser.add_argument('--gdc_method', type=str, default='ppr', help="ppr, heat, coeff")
  parser.add_argument('--gdc_sparsification', type=str, default='topk', help="threshold, topk")
  parser.add_argument('--gdc_k', type=int, default=64, help="number of neighbours to sparsify to when using topk")
  parser.add_argument('--gdc_threshold', type=float, default=0.0001,
                      help="obove this edge weight, keep edges when using threshold")
  parser.add_argument('--gdc_avg_degree', type=int, default=64,
                      help="if gdc_threshold is not given can be calculated by specifying avg degree")
  parser.add_argument('--ppr_alpha', type=float, default=0.05, help="teleport probability")
  parser.add_argument('--heat_time', type=float, default=3., help="time to run gdc heat kernal diffusion for")
  parser.add_argument('--att_samp_pct', type=float, default=1,
                      help="float in [0,1). The percentage of edges to retain based on attention scores")
  parser.add_argument('--use_flux', dest='use_flux', action='store_true',
                      help='incorporate the feature grad in attention based edge dropout')
  parser.add_argument("--exact", action="store_true",
                      help="for small datasets can do exact diffusion. If dataset is too big for matrix inversion then you can't")
  parser.add_argument('--M_nodes', type=int, default=64, help="new number of nodes to add")
  parser.add_argument('--new_edges', type=str, default="random", help="random, random_walk, k_hop")
  parser.add_argument('--sparsify', type=str, default="S_hat", help="S_hat, recalc_att")
  parser.add_argument('--threshold_type', type=str, default="topk_adj", help="topk_adj, addD_rvR")
  parser.add_argument('--rw_addD', type=float, default=0.02, help="percentage of new edges to add")
  parser.add_argument('--rw_rmvR', type=float, default=0.02, help="percentage of edges to remove")
  parser.add_argument('--rewire_KNN', action='store_true', help='perform KNN rewiring every few epochs')
  parser.add_argument('--rewire_KNN_T', type=str, default="T0", help="T0, TN")
  parser.add_argument('--rewire_KNN_epoch', type=int, default=5, help="frequency of epochs to rewire")
  parser.add_argument('--rewire_KNN_k', type=int, default=64, help="target degree for KNN rewire")
  parser.add_argument('--rewire_KNN_sym', action='store_true', help='make KNN symmetric')
  parser.add_argument('--KNN_online', action='store_true', help='perform rewiring online')
  parser.add_argument('--KNN_online_reps', type=int, default=4, help="how many online KNN its")
  parser.add_argument('--KNN_space', type=str, default="pos_distance", help="Z,P,QKZ,QKp")
  # beltrami args
  parser.add_argument('--beltrami', action='store_true', help='perform diffusion beltrami style')
  parser.add_argument('--fa_layer', action='store_true', help='add a bottleneck paper style layer with more edges')
  parser.add_argument('--pos_enc_type', type=str, default="DW64", help='positional encoder either GDC, DW64, DW128, DW256')
  parser.add_argument('--pos_enc_orientation', type=str, default="row", help="row, col")
  parser.add_argument('--feat_hidden_dim', type=int, default=64, help="dimension of features in beltrami")
  parser.add_argument('--pos_enc_hidden_dim', type=int, default=32, help="dimension of position in beltrami")
  parser.add_argument('--edge_sampling', action='store_true', help='perform edge sampling rewiring')
  parser.add_argument('--edge_sampling_T', type=str, default="T0", help="T0, TN")
  parser.add_argument('--edge_sampling_epoch', type=int, default=5, help="frequency of epochs to rewire")
  parser.add_argument('--edge_sampling_add', type=float, default=0.64, help="percentage of new edges to add")
  parser.add_argument('--edge_sampling_add_type', type=str, default="importance", help="random, ,anchored, importance, degree")
  parser.add_argument('--edge_sampling_rmv', type=float, default=0.32, help="percentage of edges to remove")
  parser.add_argument('--edge_sampling_sym', action='store_true', help='make KNN symmetric')
  parser.add_argument('--edge_sampling_online', action='store_true', help='perform rewiring online')
  parser.add_argument('--edge_sampling_online_reps', type=int, default=4, help="how many online KNN its")
  parser.add_argument('--edge_sampling_space', type=str, default="attention", help="attention,pos_distance, z_distance, pos_distance_QK, z_distance_QK")
  parser.add_argument('--symmetric_attention', action='store_true', help='maks the attention symmetric for rewring in QK space')


  parser.add_argument('--fa_layer_edge_sampling_rmv', type=float, default=0.8, help="percentage of edges to remove")
  parser.add_argument('--gpu', type=int, default=0, help="GPU to run on (default 0)")
  parser.add_argument('--pos_enc_csv', action='store_true', help="Generate pos encoding as a sparse CSV")

  parser.add_argument('--pos_dist_quantile', type=float, default=0.001, help="percentage of N**2 edges to keep")


  args = parser.parse_args()

  opt = vars(args)
  opt['is_webKB'] = True
#   print(opt["attention_type"])
#   print(opt['epoch'])
  for split in range(10):
      main(opt, split)