import argparse
import os
import time
from functools import partial

import numpy as np
import torch
from data import get_dataset, set_train_val_test_split
from GNN_early import GNNEarly
from GNN import GNN
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.suggest.ax import AxSearch
from run_GNN import get_optimizer, print_model_params, test, train
from torch import nn
from GNN_ICML import ICML_GNN, get_sym_adj
from GNN_ICML import train as train_icml

def average_test(models, datas):
  results = [test(model, data) for model, data in zip(models, datas)]
  train_accs, val_accs, tmp_test_accs = [], [], []

  for train_acc, val_acc, test_acc in results:
    train_accs.append(train_acc)
    val_accs.append(val_acc)
    tmp_test_accs.append(test_acc)

  return train_accs, val_accs, tmp_test_accs


def train_ray_rand(opt, checkpoint_dir=None, data_dir="../data", opt_val=True):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  dataset = get_dataset(opt, data_dir, True)
  
  models = []
  datas = []
  optimizers = []

  for split in range(opt["num_splits"]):
    dataset.data = set_train_val_test_split(
      np.random.randint(0, 1000), dataset.data, num_development = 5000 if opt["dataset"] == "CoauthorCS" else 1500)
    datas.append(dataset.data)

    if opt['baseline']:
      opt['num_feature'] = dataset.num_node_features
      opt['num_class'] = dataset.num_classes
      adj = get_sym_adj(dataset.data, opt, device)
      model, data = ICML_GNN(opt, adj, opt['time'], device).to(device), dataset.data.to(device)
      train_this = train_icml
    else:
      model = GNN(opt, dataset, device)
      train_this = train

    models.append(model)

    if torch.cuda.device_count() > 1:
      model = nn.DataParallel(model)

    model, data = model.to(device), dataset.data.to(device)
    parameters = [p for p in model.parameters() if p.requires_grad]

    optimizer = get_optimizer(opt["optimizer"], parameters, lr=opt["lr"], weight_decay=opt["decay"])
    optimizers.append(optimizer)

    # The `checkpoint_dir` parameter gets passed by Ray Tune when a checkpoint
    # should be restored.
    if checkpoint_dir:
      checkpoint = os.path.join(checkpoint_dir, "checkpoint")
      model_state, optimizer_state = torch.load(checkpoint)
      model.load_state_dict(model_state)
      optimizer.load_state_dict(optimizer_state)

  for epoch in range(1, opt["epoch"]):
    loss = np.mean([train_this(model, optimizer, data) for model, optimizer, data in zip(models, optimizers, datas)])
    train_accs, val_accs, tmp_test_accs = average_test(models, datas)
    with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
      best = np.argmax(val_accs)
      path = os.path.join(checkpoint_dir, "checkpoint")
      torch.save((models[best].state_dict(), optimizers[best].state_dict()), path)
    if opt_val:
      tune.report(loss=loss, accuracy=np.mean(val_accs))
    else:
      tune.report(loss=loss, accuracy=np.mean(tmp_test_accs))


def train_ray(opt, checkpoint_dir=None, data_dir="../data", opt_val=True):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  dataset = get_dataset(opt, data_dir, True)

  models = []
  optimizers = []

  data = dataset.data.to(device)
  datas = [data for i in range(opt["num_init"])]

  for split in range(opt["num_init"]):
    if opt['baseline']:
      opt['num_feature'] = dataset.num_node_features
      opt['num_class'] = dataset.num_classes
      adj = get_sym_adj(dataset.data, opt, device)
      model, data = ICML_GNN(opt, adj, opt['time'], device).to(device), dataset.data.to(device)
      train_this = train_icml
    else:
      model = GNN(opt, dataset, device)
      train_this = train

    models.append(model)

    if torch.cuda.device_count() > 1:
      model = nn.DataParallel(model)

    model = model.to(device)
    parameters = [p for p in model.parameters() if p.requires_grad]

    optimizer = get_optimizer(opt["optimizer"], parameters, lr=opt["lr"], weight_decay=opt["decay"])
    optimizers.append(optimizer)

    # The `checkpoint_dir` parameter gets passed by Ray Tune when a checkpoint
    # should be restored.
    if checkpoint_dir:
      checkpoint = os.path.join(checkpoint_dir, "checkpoint")
      model_state, optimizer_state = torch.load(checkpoint)
      model.load_state_dict(model_state)
      optimizer.load_state_dict(optimizer_state)

  for epoch in range(1, opt["epoch"]):
    loss = np.mean([train_this(model, optimizer, data) for model, optimizer in zip(models, optimizers)])
    train_accs, val_accs, tmp_test_accs = average_test(models, datas)
    with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
      best = np.argmax(val_accs)
      path = os.path.join(checkpoint_dir, "checkpoint")
      torch.save((models[best].state_dict(), optimizers[best].state_dict()), path)
    if opt_val:
      tune.report(loss=loss, accuracy=np.mean(val_accs))
    else:
      tune.report(loss=loss, accuracy=np.mean(tmp_test_accs))


def train_ray_old(opt, checkpoint_dir=None, data_dir="../data", opt_val=False):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  dataset = get_dataset(opt, data_dir, True)
  model = GNN(opt, dataset, device)
  if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
  model, data = model.to(device), dataset.data.to(device)
  parameters = [p for p in model.parameters() if p.requires_grad]
  optimizer = get_optimizer(opt["optimizer"], parameters, lr=opt["lr"], weight_decay=opt["decay"])
  # The `checkpoint_dir` parameter gets passed by Ray Tune when a checkpoint
  # should be restored.
  if checkpoint_dir:
    checkpoint = os.path.join(checkpoint_dir, "checkpoint")
    model_state, optimizer_state = torch.load(checkpoint)
    model.load_state_dict(model_state)
    optimizer.load_state_dict(optimizer_state)

  for epoch in range(1, opt["epoch"]):
    loss = train(model, optimizer, data)
    train_acc, val_acc, tmp_test_acc = test(model, data)
    with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
      path = os.path.join(checkpoint_dir, "checkpoint")
      torch.save((model.state_dict(), optimizer.state_dict()), path)
    if opt_val:
      tune.report(loss=loss, accuracy=val_acc)
    else:
      tune.report(loss=loss, accuracy=tmp_test_acc)


def train_ray_int(opt, checkpoint_dir=None, data_dir="../data", opt_val=False):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  dataset = get_dataset(opt, data_dir, True)

  if opt["num_splits"] > 0:
    dataset.data = set_train_val_test_split(
      23 * np.random.randint(0, opt["num_splits"]),  # random prime 23 to make the splits 'more' random. Could remove
      dataset.data, 
      num_development = 5000 if opt["dataset"] == "CoauthorCS" else 1500)

  model = GNN(opt, dataset, device) if opt["no_early"] else GNNEarly(opt, dataset, device)
  if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
  model, data = model.to(device), dataset.data.to(device)
  parameters = [p for p in model.parameters() if p.requires_grad]
  optimizer = get_optimizer(opt["optimizer"], parameters, lr=opt["lr"], weight_decay=opt["decay"])

  if checkpoint_dir:
    checkpoint = os.path.join(checkpoint_dir, "checkpoint")
    model_state, optimizer_state = torch.load(checkpoint)
    model.load_state_dict(model_state)
    optimizer.load_state_dict(optimizer_state)

  for epoch in range(1, opt["epoch"]):
    loss = train(model, optimizer, data)
    # need next line as it sets the attributes in the solver
    
    if opt["no_early"]:
      _, val_acc_int, tmp_test_acc_int = test(model, data)
    else:
      _, _, _ = test(model, data)
      val_acc_int = model.odeblock.test_integrator.solver.best_val
      tmp_test_acc_int = model.odeblock.test_integrator.solver.best_test
    with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
      path = os.path.join(checkpoint_dir, "checkpoint")
      torch.save((model.state_dict(), optimizer.state_dict()), path)
    if opt_val:
      tune.report(loss=loss, accuracy=val_acc_int)
    else:
      tune.report(loss=loss, accuracy=tmp_test_acc_int)


def set_cora_search_space(opt):
  opt["decay"] = tune.uniform(0.01, 0.1)  # weight decay l2 reg
  opt["kinetic_energy"]  = tune.loguniform(0.001, 10.0)
  opt["directional_penalty"]  = tune.loguniform(0.001, 10.0)

  opt["hidden_dim"] = tune.sample_from(lambda _: 2 ** np.random.randint(6, 8))  # hidden dim of X in dX/dt
  opt["lr"] = tune.loguniform(0.05, 0.2)
  opt["input_dropout"] = tune.uniform(0.2, 0.8)  # encoder dropout
  opt["optimizer"] = tune.choice(["adam", "adamax"])
  opt["dropout"] = tune.uniform(0, 0.15)  # output dropout
  opt["time"] = tune.uniform(5.0, 20.0)  # terminal time of the ODE integrator;
  # when it's big, the training hangs (probably due a big NFEs of the ODE)

  if opt["block"] in {'attention', 'mixed'} or opt['function'] in {'GAT', 'transformer', 'dorsey'}:
    opt["heads"] = tune.sample_from(lambda _: 2 ** np.random.randint(0, 4))  #
    opt["attention_dim"] = tune.sample_from(lambda _: 2 ** np.random.randint(4, 8))  # hidden dim for attention
    opt['attention_norm_idx'] = tune.choice([0, 1])
    opt["leaky_relu_slope"] = tune.uniform(0, 0.7)
    opt["self_loop_weight"] = tune.choice([0, 1])  # whether or not to use self-loops
  else:
    opt["self_loop_weight"] = tune.uniform(0, 3)

  opt["tol_scale"] = tune.loguniform(1, 1000)  # num you multiply the default rtol and atol by
  if opt["adjoint"]:
    opt["adjoint_method"] = tune.choice(["dopri5", "adaptive_heun"]) #, "rk4"])
    opt["tol_scale_adjoint"] = tune.loguniform(100, 10000)

  if opt['rewiring'] == 'gdc':
    opt['gdc_k'] = tune.sample_from(lambda _: 2 ** np.random.randint(4, 8))
    opt['ppr_alpha'] = tune.uniform(0.01, 0.2)

  return opt


def set_pubmed_search_space(opt):
  opt["decay"] = tune.uniform(0.001, 0.1)
  opt["kinetic_energy"]  = tune.loguniform(0.01, 1.0)
  opt["directional_penalty"]  = tune.loguniform(0.01, 1.0)

  opt["hidden_dim"] = 128 #tune.sample_from(lambda _: 2 ** np.random.randint(4, 8))
  opt["lr"] = tune.loguniform(0.02, 0.1)
  opt["input_dropout"] = 0.4 #tune.uniform(0.2, 0.5)
  opt["dropout"] = tune.uniform(0, 0.5)
  opt["time"] = tune.uniform(5.0, 20.0)
  opt["optimizer"] = tune.choice(["rmsprop", "adam", "adamax"])

  if opt["block"] in {'attention', 'mixed'} or opt['function'] in {'GAT', 'transformer', 'dorsey'}:
    opt["heads"] = tune.sample_from(lambda _: 2 ** np.random.randint(0, 4))
    opt["attention_dim"] = tune.sample_from(lambda _: 2 ** np.random.randint(4, 8))
    opt['attention_norm_idx'] = tune.choice([0, 1])
    opt["leaky_relu_slope"] = tune.uniform(0, 0.8)
    opt["self_loop_weight"] = tune.choice([0, 0.5, 1, 2]) if opt['block'] == 'mixed' else tune.choice(
      [0, 1])  # whether or not to use self-loops
  else:
    opt["self_loop_weight"] = tune.uniform(0, 3)

  opt["tol_scale"] = tune.loguniform(1, 1e4)

  if opt["adjoint"]:
    opt["tol_scale_adjoint"] = tune.loguniform(1, 1e4)
    opt["adjoint_method"] = tune.choice(["dopri5", "adaptive_heun"])
  else:
    raise Exception("Can't train on PubMed without the adjoint method.")

  return opt


def set_citeseer_search_space(opt):
  opt["decay"] = 0.1 #tune.loguniform(2e-3, 1e-2)
  opt["kinetic_energy"]  = tune.loguniform(0.001, 10.0)
  opt["directional_penalty"]  = tune.loguniform(0.001, 10.0)

  opt["hidden_dim"] = 128 #tune.sample_from(lambda _: 2 ** np.random.randint(6, 8))
  opt["lr"] = tune.loguniform(2e-3, 0.01)
  opt["input_dropout"] = tune.uniform(0.4, 0.8)
  opt["dropout"] = tune.uniform(0, 0.8)
  opt["time"] = tune.uniform(0.5, 8.0)
  opt["optimizer"] = tune.choice(["rmsprop", "adam", "adamax"])
  #

  if opt["block"] in {'attention', 'mixed'} or opt['function'] in {'GAT', 'transformer', 'dorsey'}:
    opt["heads"] = tune.sample_from(lambda _: 2 ** np.random.randint(1, 4))
    opt["attention_dim"] = tune.sample_from(lambda _: 2 ** np.random.randint(3, 8))
    opt['attention_norm_idx'] = 1  # tune.choice([0, 1])
    opt["leaky_relu_slope"] = tune.uniform(0, 0.7)
    opt["self_loop_weight"] = tune.choice([0, 0.5, 1, 2]) if opt['block'] == 'mixed' else tune.choice(
      [0, 1])  # whether or not to use self-loops
  else:
    opt["self_loop_weight"] = tune.uniform(0, 3)  # 1 seems to work pretty well

  opt["tol_scale"] = tune.loguniform(1, 2e3)

  if opt["adjoint"]:
    opt["tol_scale_adjoint"] = tune.loguniform(1, 1e5)
    opt["adjoint_method"] = tune.choice(["dopri5", "adaptive_heun"])  # , "rk4"])
  return opt


def set_computers_search_space(opt):
  opt["decay"] = tune.loguniform(2e-3, 1e-2)
  opt["kinetic_energy"]  = tune.loguniform(0.01, 10.0)
  opt["directional_penalty"]  = tune.loguniform(0.001, 10.0)

  opt["hidden_dim"] = tune.sample_from(lambda _: 2 ** np.random.randint(4, 8))
  opt["lr"] = tune.loguniform(5e-5, 5e-3)
  opt["input_dropout"] = tune.uniform(0.4, 0.8)
  opt["dropout"] = tune.uniform(0, 0.8)
  opt["self_loop_weight"] = tune.choice([0, 1])
  opt["time"] = tune.uniform(0.5, 10.0)
  opt["optimizer"] = tune.choice(["adam", "adamax", "rmsprop"])

  if opt["block"] in {'attention', 'mixed'} or opt['function'] in {'GAT', 'transformer', 'dorsey'}:
    opt["heads"] = tune.sample_from(lambda _: 2 ** np.random.randint(0, 4))
    opt["attention_dim"] = tune.sample_from(lambda _: 2 ** np.random.randint(3, 8))
    opt['attention_norm_idx'] = 1  # tune.choice([0, 1])
    opt["leaky_relu_slope"] = tune.uniform(0, 0.8)
    opt["self_loop_weight"] = tune.choice([0, 0.5, 1, 2]) if opt['block'] == 'mixed' else tune.choice(
      [0, 1])  # whether or not to use self-loops
  else:
    opt["self_loop_weight"] = tune.uniform(0, 3)

  opt["tol_scale"] = tune.loguniform(1e1, 1e4)

  if opt["adjoint"]:
    opt["tol_scale_adjoint"] = tune.loguniform(1, 1e5)
    opt["adjoint_method"] = tune.choice(["dopri5", "adaptive_heun", "rk4"])
  return opt


def set_coauthors_search_space(opt):
  opt["decay"] = tune.loguniform(2e-3, 1e-2)
  opt["kinetic_energy"]  = tune.loguniform(0.01, 10.0)
  opt["directional_penalty"]  = tune.loguniform(0.01, 10.0)

  opt["hidden_dim"] = tune.sample_from(lambda _: 2 ** np.random.randint(4, 6))
  opt["lr"] = tune.loguniform(1e-5, 0.1)
  opt["input_dropout"] = tune.uniform(0.4, 0.8)
  opt["dropout"] = tune.uniform(0, 0.8)
  opt["self_loop_weight"] = tune.choice([0, 1])
  opt["time"] = tune.uniform(0.5, 10.0)
  opt["optimizer"] = tune.choice(["adam", "adamax", "rmsprop"])

  if opt["block"] in {'attention', 'mixed'} or opt['function'] in {'GAT', 'transformer', 'dorsey'}:
    opt["heads"] = tune.sample_from(lambda _: 2 ** np.random.randint(0, 4))
    opt["attention_dim"] = tune.sample_from(lambda _: 2 ** np.random.randint(3, 8))
    opt['attention_norm_idx'] = tune.choice([0, 1])
    opt["leaky_relu_slope"] = tune.uniform(0, 0.8)
    opt["self_loop_weight"] = tune.choice([0, 0.5, 1, 2]) if opt['block'] == 'mixed' else tune.choice(
      [0, 1])  # whether or not to use self-loops
  else:
    opt["self_loop_weight"] = tune.uniform(0, 3)

  opt["tol_scale"] = tune.loguniform(1e1, 1e4)

  if opt["adjoint"]:
    opt["tol_scale_adjoint"] = tune.loguniform(1, 1e5)
    opt["adjoint_method"] = tune.choice(["dopri5", "adaptive_heun", "rk4"])
  return opt


def set_photo_search_space(opt):
  opt["decay"] = tune.loguniform(2e-3, 1e-2)
  opt["kinetic_energy"]  = tune.loguniform(0.01, 10.0)
  opt["directional_penalty"]  = tune.loguniform(0.001, 10.0)

  opt["hidden_dim"] = tune.sample_from(lambda _: 2 ** np.random.randint(3, 7))
  opt["lr"] = tune.loguniform(1e-2, 0.1)
  opt["input_dropout"] = tune.uniform(0.4, 0.8)
  opt["dropout"] = tune.uniform(0, 0.8)
  opt["time"] = tune.uniform(0.5, 7.0)
  opt["optimizer"] = tune.choice(["adam", "adamax", "rmsprop"])

  if opt["block"] in {'attention', 'mixed'} or opt['function'] in {'GAT', 'transformer', 'dorsey'}:
    opt["heads"] = tune.sample_from(lambda _: 2 ** np.random.randint(0, 3))
    opt["attention_dim"] = tune.sample_from(lambda _: 2 ** np.random.randint(3, 6))
    opt['attention_norm_idx'] = tune.choice([0, 1])
    opt["self_loop_weight"] = tune.choice([0, 0.5, 1, 2]) if opt['block'] == 'mixed' else tune.choice(
      [0, 1]) 
    opt["leaky_relu_slope"] = tune.uniform(0, 0.8)
  else: 
    opt["self_loop_weight"] = tune.uniform(0, 3)

  opt["tol_scale"] = tune.loguniform(1, 1e4)

  if opt["adjoint"]:
    opt["tol_scale_adjoint"] = tune.loguniform(1, 1e5)
    opt["adjoint_method"] = tune.choice(["dopri5", "adaptive_heun", "rk4"])
  return opt


def set_search_space(opt):
  if opt["dataset"] == "Cora":
    return set_cora_search_space(opt)
  elif opt["dataset"] == "Pubmed":
    return set_pubmed_search_space(opt)
  elif opt["dataset"] == "Citeseer":
    return set_citeseer_search_space(opt)
  elif opt["dataset"] == "Computers":
    return set_computers_search_space(opt)
  elif opt["dataset"] == "Photo":
    return set_photo_search_space(opt)
  elif opt["dataset"] == "CoauthorCS":
    return set_coauthors_search_space(opt)


def main(opt):
  data_dir = os.path.abspath("../data")
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  opt = set_search_space(opt)
  scheduler = ASHAScheduler(
    metric="accuracy",
    mode="max",
    max_t=opt["epoch"],
    grace_period=opt["grace_period"],
    reduction_factor=opt["reduction_factor"],
  )
  reporter = CLIReporter(
    metric_columns=["accuracy", "loss", "training_iteration"]
  )
  # choose a search algorithm from https://docs.ray.io/en/latest/tune/api_docs/suggestion.html
  search_alg = AxSearch(metric="accuracy")
  search_alg = None

  train_fn = train_ray if opt["num_splits"] == 0 else train_ray_rand

  result = tune.run(
    partial(train_fn, data_dir=data_dir),
    name=opt["name"],
    resources_per_trial={"cpu": opt["cpus"], "gpu": opt["gpus"]},
    search_alg=search_alg,
    config=opt,
    num_samples=opt["num_samples"],
    scheduler=scheduler,
    max_failures=2,
    local_dir="../ray_tune",
    progress_reporter=reporter,
    raise_on_failed_trial=False,
  )

  # df = result.dataframe(metric='accuracy', mode='max')
  best_trial = result.get_best_trial("accuracy", "max", "all")
  print("Best trial config: {}".format(best_trial.config))
  print("Best trial final validation loss: {}".format(best_trial.best_result["loss"]))
  print("Best trial final validation accuracy: {}".format(best_trial.best_result["accuracy"]))

  dataset = get_dataset(opt, data_dir, True)
  best_trained_model = GNN(best_trial.config, dataset, device)
  if opt["gpus"] > 1:
    best_trained_model = nn.DataParallel(best_trained_model)
  best_trained_model.to(device)

  checkpoint_path = os.path.join(best_trial.checkpoint.value, "checkpoint")

  model_state, optimizer_state = torch.load(checkpoint_path)
  best_trained_model.load_state_dict(model_state)

  test_acc = test(best_trained_model, best_trained_model.data.to(device))
  print("Best trial test set accuracy: {}".format(test_acc))
  df = result.dataframe(metric="accuracy", mode="max").sort_values(
    "accuracy", ascending=False
  )  # get max accuracy for each trial
  timestr = time.strftime("%Y%m%d-%H%M%S")
  df.to_csv("../hyperopt_results/result_{}.csv".format(timestr))


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--use_cora_defaults",
    action="store_true",
    help="Whether to run with best params for cora. Overrides the choice of dataset",
  )
  parser.add_argument(
    "--dataset", type=str, default="Cora", help="Cora, Citeseer, Pubmed, Computers, Photo, CoauthorCS"
  )
  parser.add_argument("--hidden_dim", type=int, default=32, help="Hidden dimension.")
  parser.add_argument("--input_dropout", type=float, default=0.5, help="Input dropout rate.")
  parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate.")
  parser.add_argument("--optimizer", type=str, default="adam", help="Optimizer.")
  parser.add_argument("--lr", type=float, default=0.01, help="Learning rate.")
  parser.add_argument("--decay", type=float, default=5e-4, help="Weight decay for optimization")
  parser.add_argument("--self_loop_weight", type=float, default=1.0, help="Weight of self-loops.")
  parser.add_argument("--epoch", type=int, default=10, help="Number of training epochs per iteration.")
  parser.add_argument("--alpha", type=float, default=1.0, help="Factor in front matrix A.")
  parser.add_argument("--time", type=float, default=1.0, help="End time of ODE function.")
  parser.add_argument("--augment", action="store_true",
                      help="double the length of the feature vector by appending zeros to stabilise ODE learning", )
  parser.add_argument("--alpha_dim", type=str, default="sc", help="choose either scalar (sc) or vector (vc) alpha")
  parser.add_argument("--alpha_sigmoid", type=bool, default=True, help="apply sigmoid before multiplying by alpha")
  parser.add_argument("--beta_dim", type=str, default="sc", help="choose either scalar (sc) or vector (vc) beta")
  # ODE args
  parser.add_argument(
    "--method", type=str, default="dopri5", help="set the numerical solver: dopri5, euler, rk4, midpoint"
  )
  parser.add_argument(
    "--adjoint_method", type=str, default="adaptive_heun",
    help="set the numerical solver for the backward pass: dopri5, euler, rk4, midpoint"
  )
  parser.add_argument("--adjoint", default=False, help="use the adjoint ODE method to reduce memory footprint")
  parser.add_argument("--tol_scale", type=float, default=1.0, help="multiplier for atol and rtol")
  parser.add_argument("--tol_scale_adjoint", type=float, default=1.0,
                      help="multiplier for adjoint_atol and adjoint_rtol")
  parser.add_argument("--ode_blocks", type=int, default=1, help="number of ode blocks to run")
  parser.add_argument(
    "--simple", type=bool, default=True, help="If try get rid of alpha param and the beta*x0 source term"
  )
  # SDE args
  parser.add_argument("--dt_min", type=float, default=1e-5, help="minimum timestep for the SDE solver")
  parser.add_argument("--dt", type=float, default=1e-3, help="fixed step size")
  parser.add_argument("--adaptive", type=bool, default=False, help="use adaptive step sizes")
  # Attention args
  parser.add_argument("--attention_dropout", type=float, default=0.0, help="dropout of attention weights")
  parser.add_argument(
    "--leaky_relu_slope",
    type=float,
    default=0.2,
    help="slope of the negative part of the leaky relu used in attention",
  )
  parser.add_argument('--attention_dim', type=int, default=64,
                      help='the size to project x to before calculating att scores')
  parser.add_argument("--heads", type=int, default=4, help="number of attention heads")
  parser.add_argument("--attention_norm_idx", type=int, default=0, help="0 = normalise rows, 1 = normalise cols")
  parser.add_argument("--mix_features", type=bool, default=False, help="False apply attention to x. True apply to xW")

  parser.add_argument('--block', type=str, default='constant', help='constant, mixed, attention, SDE')
  parser.add_argument('--function', type=str, default='laplacian', help='laplacian, transformer, dorsey, GAT, SDE')
  parser.add_argument('--reweight_attention', type=bool, default=False, help="multiply attention scores by edge weights before softmax")
  # ray args
  parser.add_argument("--num_samples", type=int, default=20, help="number of ray trials")
  parser.add_argument("--gpus", type=float, default=0, help="number of gpus per trial. Can be fractional")
  parser.add_argument("--cpus", type=float, default=1, help="number of cpus per trial. Can be fractional")
  parser.add_argument(
    "--grace_period", type=int, default=5, help="number of epochs to wait before terminating trials"
  )
  parser.add_argument(
    "--reduction_factor", type=int, default=4, help="number of trials is halved after this many epochs"
  )
  parser.add_argument("--name", type=str, default="ray_exp")
  parser.add_argument("--num_splits", type=int, default=0, help="Number of random splits >= 0. 0 for planetoid split")
  parser.add_argument("--num_init", type=int, default=4, help="Number of random initializations >= 0")

  parser.add_argument("--max_nfe", type=int, default=300, help="Maximum number of function evaluations allowed.")

  parser.add_argument('--jacobian_norm2', type=float, default=None, help="int_t ||df/dx||_F^2")
  parser.add_argument('--total_deriv', type=float, default=None, help="int_t ||df/dt||^2")

  parser.add_argument('--kinetic_energy', type=float, default=None, help="int_t ||f||_2^2")
  parser.add_argument('--directional_penalty', type=float, default=None, help="int_t ||(df/dx)^T f||^2")

  parser.add_argument("--baseline", action="store_true", help="Wheather to run the ICML baseline or not.")

  # rewiring args
  parser.add_argument('--rewiring', type=str, default=None, help="two_hop, gdc")
  parser.add_argument('--gdc_method', type=str, default='ppr', help="ppr, heat, coeff")
  parser.add_argument('--gdc_sparsification', type=str, default='topk', help="threshold, topk")
  parser.add_argument('--gdc_k', type=int, default=64, help="number of neighbours to sparsify to when using topk")
  parser.add_argument('--gdc_threshold', type=float, default=0.0001, help="obove this edge weight, keep edges when using threshold")
  parser.add_argument('--gdc_avg_degree', type=int, default=64,
                      help="if gdc_threshold is not given can be calculated by specifying avg degree")
  parser.add_argument('--ppr_alpha', type=float, default=0.05, help="teleport probability")
  parser.add_argument('--heat_time', type=float, default=3., help="time to run gdc heat kernal diffusion for")

  args = parser.parse_args()

  opt = vars(args)

  main(opt)
