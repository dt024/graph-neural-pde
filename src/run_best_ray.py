import argparse
from ray.tune import Analysis
import json
import numpy as np
from utils import get_sem, mean_confidence_interval
from ray_tune import train_ray_int
from ray import tune
from functools import partial
import os, time
from ray.tune import CLIReporter


def get_best_params_dir(opt):
  analysis = Analysis("../ray_tune/{}".format(opt['folder']))
  df = analysis.dataframe(metric=opt['metric'], mode='max')
  best_params_dir = df.sort_values('accuracy', ascending=False)['logdir'].iloc[opt['index']]
  return best_params_dir


def run_best_params(opt):
  best_params_dir = get_best_params_dir(opt)
  with open(best_params_dir + '/params.json') as f:
    best_params = json.loads(f.read())
  # allow params specified at the cmd line to override
  best_params_ret = {**best_params, **opt}
  try:
    best_params_ret['mix_features']
  except KeyError:
    best_params_ret['mix_features'] = False
  try:
    best_params_ret['fc_out']
  except KeyError:
    best_params_ret['fc_out'] = False
  try:
    best_params_ret['kinetic_energy']
  except KeyError:
    best_params_ret['kinetic_energy'] = None
  try:
    best_params_ret['jacobian_norm2']
  except KeyError:
    best_params_ret['jacobian_norm2'] = None
  try:
    best_params_ret['total_deriv']
  except KeyError:
    best_params_ret['total_deriv'] = None
  try:
    best_params_ret['directional_penalty']
  except KeyError:
    best_params_ret['directional_penalty'] = None
  try:
    best_params_ret['data_norm']
  except KeyError:
    best_params_ret['data_norm'] = 'gcn'
  try:
    best_params_ret['batch_norm']
  except KeyError:
    best_params_ret['batch_norm'] = False
  try:
    best_params_ret['method']
  except KeyError:
    best_params_ret['method'] = 'dopri5'
  try:
    best_params_ret['step_size']
  except KeyError:
    best_params_ret['step_size'] = 1
  try:
    best_params_ret['adjoint_step_size']
  except KeyError:
    best_params_ret['adjoint_step_size'] = 1
  try:
    best_params_ret['max_iters']
  except KeyError:
    best_params_ret['max_iters'] = 100
  try:
    best_params_ret['no_alpha_sigmoid']
  except KeyError:
    best_params_ret['no_alpha_sigmoid'] = False
  try:
    best_params_ret['add_source']
  except KeyError:
    best_params_ret['add_source'] = False
  try:
    best_params_ret['tol_scale']
  except KeyError:
    best_params_ret['tol_scale'] = 1
  try:
    best_params_ret['tol_scale_adjoint']
  except KeyError:
    best_params_ret['tol_scale_adjoint'] = 1

  # the exception is number of epochs as we want to use more here than we would for hyperparameter tuning.
  best_params_ret['epoch'] = opt['epoch']
  best_params_ret['max_nfe'] = opt['max_nfe']
  # handle adjoint
  if best_params['adjoint'] or opt['adjoint']:
    best_params_ret['adjoint'] = True
  # handle labels
  if best_params['use_labels'] or opt['use_labels']:
    best_params_ret['adjoint'] = True

  print("Running with parameters {}".format(best_params_ret))

  data_dir = os.path.abspath("../data")
  reporter = CLIReporter(
    metric_columns=["accuracy", "loss", "test_acc", "train_acc", "best_time", "best_epoch", "training_iteration", "forward_nfe", "backward_nfe"])

  if opt['name'] is None:
    name = opt['folder'] + '_test'
  else:
    name = opt['name']

  result = tune.run(
    partial(train_ray_int, data_dir=data_dir),
    name=name,
    resources_per_trial={"cpu": opt['cpus'], "gpu": opt['gpus']},
    search_alg=None,
    keep_checkpoints_num=3,
    checkpoint_score_attr='accuracy',
    config=best_params_ret,
    num_samples=opt['reps'] if opt["num_splits"] == 0 else opt["num_splits"] * opt["reps"],
    scheduler=None,
    max_failures=1,  # early stop solver can't recover from failure as it doesn't own m2.
    local_dir='../ray_tune',
    progress_reporter=reporter,
    raise_on_failed_trial=False)


  df = result.dataframe(metric=opt['metric'], mode="max").sort_values(opt['metric'], ascending=False)

  print(df[['accuracy', 'test_acc', 'train_acc', 'best_time', 'best_epoch']])

  test_accs = df['test_acc'].values
  print("test accuracy {}".format(test_accs))
  log = "mean test {:04f}, test std {:04f}, test sem {:04f}, test 95% conf {:04f}"
  print(log.format(test_accs.mean(), np.std(test_accs), get_sem(test_accs), mean_confidence_interval(test_accs)))

  df.to_csv('../ray_results/{}_{}.csv'.format(name, time.strftime("%Y%m%d-%H%M%S")))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--epoch', type=int, default=10, help='Number of training epochs per iteration.')
  parser.add_argument('--folder', type=str, default=None, help='experiment folder to read')
  parser.add_argument('--index', type=int, default=0, help='index to take from experiment folder')
  parser.add_argument('--metric', type=str, default='accuracy', help='metric to sort the hyperparameter tuning runs on')
  parser.add_argument('--augment', action='store_true',
                      help='double the length of the feature vector by appending zeros to stabilise ODE learning')
  parser.add_argument('--reps', type=int, default=1, help='the number of random weight initialisations to use')
  parser.add_argument('--name', type=str, default=None)
  parser.add_argument('--gpus', type=float, default=0, help='number of gpus per trial. Can be fractional')
  parser.add_argument('--cpus', type=float, default=1, help='number of cpus per trial. Can be fractional')
  parser.add_argument("--num_splits", type=int, default=0, help="Number of random slpits >= 0. 0 for planetoid split")
  parser.add_argument("--adjoint", dest='adjoint', action='store_true',
                      help="use the adjoint ODE method to reduce memory footprint")
  parser.add_argument("--max_nfe", type=int, default=5000, help="Maximum number of function evaluations allowed.")
  parser.add_argument("--no_early", action="store_true",
                      help="Whether or not to use early stopping of the ODE integrator when testing.")
  parser.add_argument('--use_labels', dest='use_labels', action='store_true', help='Also diffuse labels')

  parser.add_argument('--earlystopxT', type=float, default=3, help='multiplier for T used to evaluate best model')

  args = parser.parse_args()

  opt = vars(args)
  run_best_params(opt)
