"""
Implementation of the functions proposed in Graph embedding energies and diffusion
"""

import torch
from torch import nn
import torch_sparse
from torch_scatter import scatter_add, scatter_mul
from torch_geometric.utils.loop import add_remaining_self_loops
from torch_geometric.nn.inits import glorot, zeros
from torch.nn import Parameter

from utils import MaxNFEException
from base_classes import ODEFunc

# REMOVE SOURCES OF RANDOMNESS
import numpy as np
import random

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)


# torch.use_deterministic_algorithms(True)

class ODEFuncGreed(ODEFunc):

  def __init__(self, in_features, out_features, opt, data, device, bias=False):
    super(ODEFuncGreed, self).__init__(opt, data, device)
    assert opt['self_loop_weight'] == 0, 'greed does not work with self-loops as eta becomes zero everywhere'
    self.in_features = in_features
    self.out_features = out_features
    if opt['self_loop_weight'] > 0:
      self.edge_index, self.edge_weight = add_remaining_self_loops(data.edge_index, data.edge_attr,
                                                                   fill_value=opt['self_loop_weight'])
    else:
      self.edge_index, self.edge_weight = data.edge_index, data.edge_attr

    self.n_nodes = data.x.shape[0]
    self.self_loops = self.get_self_loops()

    self.K = Parameter(torch.Tensor(out_features, 1))
    self.Q = Parameter(torch.Tensor(out_features, 1))

    self.deg_inv_sqrt = self.get_deg_inv_sqrt(data)
    self.deg_inv = self.deg_inv_sqrt * self.deg_inv_sqrt

    self.W = Parameter(torch.Tensor(in_features, opt['attention_dim']))

    if bias:
      self.bias = Parameter(torch.Tensor(out_features))
    else:
      self.register_parameter('bias', None)

    # todo chosen to balance x0 and f in the forward function. May not be optimal
    self.mu = nn.Parameter(torch.tensor(1.))
    self.alpha = nn.Parameter(torch.tensor(1.))

    self.reset_parameters()

  def reset_parameters(self):
    glorot(self.W)
    glorot(self.K)
    glorot(self.Q)
    zeros(self.bias)

  def get_deg_inv_sqrt(self, data):
    edge_index = data.edge_index
    edge_weight = torch.ones((edge_index.size(1),), dtype=data.x.dtype,
                             device=edge_index.device)
    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, row, dim=0, dim_size=self.n_nodes)
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt = deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
    return deg_inv_sqrt

  def symmetrically_normalise(self, x, edge):
    """
    symmetrically normalise a sparse matrix with values x and edges edge. Note that edge need not be the edge list
    of the input graph as we could be normalising a Laplacian, which additionally has values on the diagonal
    D^{-1/2}MD^{-1/2}
    where D is the degree matrix
    """
    assert x.shape[0] == edge.shape[
      1], "can only symmetrically normalise a sparse matrix with the same number of edges " \
          "as edge"
    row, col = edge[0], edge[1]
    dis = self.deg_inv_sqrt
    normed_x = dis[row] * x * dis[col]
    return normed_x

  def get_tau(self, x):
    """
    Tau plays a role similar to the diffusivity / attention function in BLEND. Here the choice of nonlinearity is not
    plug and play as everything is defined on the level of the energy and so the derivatives are hardcoded.
    #todo why do we hardcode the derivatives? Couldn't we just use autograd?
    @param x:
    @param edge:
    @return:
    """
    src_x, dst_x = self.get_src_dst(x)
    tau = torch.tanh(src_x @ self.K + dst_x @ self.Q)
    tau_transpose = torch.tanh(dst_x @ self.K + src_x @ self.Q)
    return tau, tau_transpose

  def get_src_dst(self, x):
    """
    Get the values of a dense n-by-d matrix
    @param x:
    @return:
    """
    src = x[self.edge_index[0, :]]
    dst = x[self.edge_index[1, :]]
    return src, dst

  def get_metric(self, x, tau, tau_transpose):
    """
    @param x:
    @param edge:
    @param tau:
    @param tau_transpose:
    @param epsilon:
    @return:
    """
    energy_gradient = self.get_energy_gradient(x, tau, tau_transpose)
    metric = torch.sum(energy_gradient * energy_gradient, dim=1)
    return metric

  def get_energy_gradient(self, x, tau, tau_transpose):
    src_x, dst_x = self.get_src_dst(x)
    src_deg_inv_sqrt, dst_deg_inv_sqrt = self.get_src_dst(self.deg_inv_sqrt)
    src_term = (tau * src_x * src_deg_inv_sqrt.unsqueeze(dim=-1))
    # src_term.masked_fill_(src_term == float('inf'), 0.)
    dst_term = (tau_transpose * dst_x * dst_deg_inv_sqrt.unsqueeze(dim=-1))
    # dst_term.masked_fill_(dst_term == float('inf'), 0.)
    # W is [d,p]
    energy_gradient = (src_term - dst_term) @ self.W
    return energy_gradient

  def get_gamma(self, metric, epsilon=1e-2):
    # todo make epsilon smaller
    # The degree is the thing that provides a notion of dimension on a graph
    # eta is the determinant of the ego graph. This should slow down diffusion where the grad is large
    row, col = self.edge_index
    # todo check if there's a function that does all of this in one go that's optimised for autograd
    log_eta = self.deg_inv * scatter_add(torch.log(metric), row)
    eta = torch.exp(log_eta)
    src_eta, dst_eta = self.get_src_dst(eta)
    gamma = -torch.div(src_eta + dst_eta, 2 * metric)
    with torch.no_grad():
      mask = metric < epsilon
      gamma[mask] = 0
    return gamma, eta

  def get_self_loops(self):
    loop_index = torch.arange(0, self.n_nodes, dtype=self.edge_index.dtype, device=self.edge_index.device)
    loop_index = loop_index.unsqueeze(0).repeat(2, 1)
    return loop_index

  def spvm(self, index, values, vector):
    row, col = index
    out = vector[col]
    out = out * values
    out = scatter_add(out, row, dim=-1)
    return out

  def get_R1(self, f, T2, T3, fWs):
    T4 = self.get_laplacian_form(T3, T2)
    edges = torch.cat([self.edge_index, self.self_loops], dim=1)
    temp = torch_sparse.spmm(edges, T4, fWs.shape[0], fWs.shape[0], fWs)
    R1 = torch.sum(f * temp, dim=1)
    return R1

  def get_R2(self, T2, T5, T5_edge_index, f, fWs):
    temp = torch_sparse.spmm(T5_edge_index, T5, fWs.shape[0], fWs.shape[0], fWs)
    term1 = torch.sum(f * temp, dim=1)
    transposed_edge_index, T2_transpose = torch_sparse.transpose(self.edge_index, T2, self.n_nodes, self.n_nodes)
    temp1 = self.deg_inv * torch.sum(f * fWs, dim=1)
    term2 = self.spvm(transposed_edge_index, T2_transpose, temp1)
    return term2 - term1

  def get_laplacian_form(self, A, D):
    """
    Takes two sparse matrices A and D and performs sym_norm(D' - A) where D' is the degree matrix from row summing D
    @param A: Matrix that plays the role of the adjacency
    @param D: Matrix that is row summed to play the role of the degree matrix
    @return: A Laplacian form
    """
    degree = scatter_add(D, self.edge_index[0, :], dim=-1, dim_size=self.n_nodes)
    edges = torch.cat([self.edge_index, self.self_loops], dim=1)
    values = torch.cat([-A, degree], dim=-1)
    L = self.symmetrically_normalise(values, edges)
    return L

  def get_dynamics(self, f, gamma, tau, tau_transpose, Ws):
    """
    These dynamics follow directly from differentiating the energy
    The structure of T0,T1 etc. is a direct result of differentiating the hyperbolic tangent nonlinearity.
    Note: changing the nonlinearity will change the equations
    Returns: L - a generalisation of the Laplacian, but using attention
             R1, R2 are a result of tau being a function of the features and both are zero if tau is not dependent on
             features
    """
    tau = tau.flatten()
    tau_transpose = tau_transpose.flatten()
    tau2 = tau * tau
    tau3 = tau * tau2
    T0 = gamma * tau2
    T1 = gamma * tau * tau_transpose
    T2 = gamma * (tau - tau3)
    T3 = gamma * (tau_transpose - tau_transpose * tau2)
    # sparse transpose changes the edge index instead of the data and this must be carried through
    transposed_edge_index, T3_transpose = torch_sparse.transpose(self.edge_index, T3, self.n_nodes, self.n_nodes)
    T5 = self.symmetrically_normalise(T3_transpose, transposed_edge_index)
    L = self.get_laplacian_form(T1, T0)
    fWs = torch.matmul(f, Ws)
    R1 = self.get_R1(f, T2, T3, fWs)
    R2 = self.get_R2(T2, T5, transposed_edge_index, f, fWs)
    return L, R1, R2

  def clipper(self, tensors, threshold=10):
    for M in tensors:
      M = M.masked_fill(M > threshold, threshold)
      M = M.masked_fill(M < -threshold, -threshold)
    return tensors

  def get_energy(self, x, eta):
    term1 = 0.5 * torch.sum(eta)
    term2 = self.mu * torch.sum((x - self.x0) ** 2)
    return term1 + term2

  def forward(self, t, x):  # t is needed when called by the integrator
    if self.nfe > self.opt["max_nfe"]:
      raise MaxNFEException
    self.nfe += 1
    Ws = self.W @ self.W.t()  # output a [d,d] tensor
    tau, tau_transpose = self.get_tau(x)
    metric = self.get_metric(x, tau, tau_transpose)
    gamma, eta = self.get_gamma(metric)
    L, R1, R2 = self.get_dynamics(x, gamma, tau, tau_transpose, Ws)
    # L, R1, R2 = self.clipper([L, R1, R2])
    edges = torch.cat([self.edge_index, self.self_loops], dim=1)
    f = torch_sparse.spmm(edges, L, x.shape[0], x.shape[0], x)
    f = torch.matmul(f, Ws)
    f = f + R1.unsqueeze(dim=-1) @ self.K.t() + R2.unsqueeze(dim=-1) @ self.Q.t()
    f = f - 0.5 * self.mu * (x - self.x0)
    # f = f + self.x0
    # todo consider adding a term f = f + self.alpha * f
    energy = self.get_energy(x, eta)
    # energy = 0.5 * torch.sum(metric) + self.mu * torch.sum((x - self.x0) ** 2)
    print(f"energy = {energy} at time {t} and mu={self.mu}")
    return f

  def __repr__(self):
    return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'