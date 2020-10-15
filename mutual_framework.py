import numpy as np 
from scipy.integrate import odeint 
import matplotlib.pyplot as plt
import time 
import os
import pandas as pd
import multiprocessing as mp
from scipy.optimize import fsolve, root
import networkx as nx
import scipy.integrate as sin
import seaborn as sns
import sympy as sp


def ode_Cheng(f, y0, tspan, *args):
    """Solve ordinary differential equation by simple integration

    :f: function that governs the deterministic part
    :y0: initial condition
    :tspan: simulation period
    :returns: solution of y 

    """
    N = len(tspan)
    d = np.size(y0)
    dt = (tspan[N-1] - tspan[0])/(N - 1)
    # allocate space for result
    y = np.zeros((N, d), dtype=type(y0[0]))
    y[0] = y0
    for n in range(N-1):
        tn = tspan[n]
        yn = y[n]
        y[n+1] = yn + f(yn, tn, *args) * dt
    return y


def load_data(net_type):
    """TODO: Docstring for gen_data.

    :arg1: TODO
    :returns: TODO

    """

    if net_type == 1 or net_type == 2:
        data = scipy.io.loadmat('/home/mac/RPI/research/unires/sim_matlab/NuRsE-master/figure2/M_real_data/ANEMONE_FISH_WEBS_Coral_reefs2007.mat')

    elif net_type == 3 or net_type == 4:
        data = scipy.io.loadmat('/home/mac/RPI/research/unires/sim_matlab/NuRsE-master/figure2/M_real_data/PLANT_ANT_WEBS_Bluthgen_2004.mat')

    elif net_type == 5 or net_type == 6:
        data = scipy.io.loadmat('/home/mac/RPI/research/unires/sim_matlab/NuRsE-master/figure2/M_real_data/PLANT_POLLINATOR_BEES_WASPS_Clements_1923.mat')

    elif net_type == 7 or net_type == 8:
        data = scipy.io.loadmat('/home/mac/RPI/research/unires/sim_matlab/NuRsE-master/figure2/M_real_data/PLANT_POLLINATOR_BEES_WASPS_Elberling_1999.mat')

    elif net_type == 9 or net_type == 10:
        data = scipy.io.loadmat('/home/mac/RPI/research/unires/sim_matlab/NuRsE-master/figure2/M_real_data/PLANT_POLLINATOR_BEES_WASPS_Santos_2010.mat')

    elif net_type == 11 or net_type == 12:
        data = scipy.io.loadmat('/home/mac/RPI/research/unires/sim_matlab/NuRsE-master/figure2/M_real_data/PLANT_POLLINATOR_Robertson_1929.mat')

    elif net_type == 13 or net_type == 14:
        data = scipy.io.loadmat('/home/mac/RPI/research/unires/sim_matlab/NuRsE-master/figure2/M_real_data/PLANT_SEED_DISPERSER_Schleuning_2010.mat')

    else:
        print('wrong network type!')
        return None

    if net_type%2 == 1:
        A = data['A']  # adjacency matrix of plant network  
        M = data['M']
        N = np.size(A, 0)

    else:
        A = data['B']  # adjacency matrix of pollinator network  
        M = data['M']
        N = np.size(A, 0)

    return A, M, N

def A_from_data(net_type, M):
    """project network from bipartite network
    :net_type: if odd, construct adjacency network from 'A'; if even, construct adjacency network from 'B'
    :M: bipartite network
    :returns: project network for plant

    """
    m, n = M.shape
    M_1mn = M.reshape(1, m, n)  # reshape M to 3D matrix 
    if net_type == 1:
        M_nm1 = np.transpose(M_1mn, (2,1,0))  # M_nm1 is n * m * 1 matrix 
        M_3d = M_nm1 + M 
        M_0 = M_nm1 * M  # if the element of M_0 is 0, there is no common species shared with two plants 
        "suppose unweighted interaction network, which means it is blind for bees to choose which plant should pollinate."
        k = M.sum(-1)  # total node weight of species in the other network B 
        A = np.sum(M_3d * np.heaviside(M_0, 0) / k.reshape(m, 1), axis=1) 
    elif net_type == 0:
        M_nm1 = np.transpose(M_1mn, (1,2,0))  # M_nm1 is n * m * 1 matrix 
        M_3d = M_nm1 + M.T
        M_0 = M_nm1 * M.T  # if the element of M_0 is 0, there is no common species shared with two plants 
        k = M.sum(0)
        A = np.sum(M_3d * np.heaviside(M_0, 0) / k.reshape(n, 1), axis=1) 

    else:
        print('wrong net type')
        return None
    np.fill_diagonal(A, 0)  # set diagonal of adjacency matrix 0 
    return A 

def Gcc_A_mat(A, initial, remove):
    """find the survive nodes which is in giant connected components for a given adjacency matrix
    
    :A: TODO
    :returns: TODO

    """
    G = nx.from_numpy_matrix(A)
    G.remove_nodes_from(remove)
    # A_update = np.delete(A_copy, rm, 0)
    # A_update = np.delete(A_update, rm, 1)
    # initial_update = np.delete(initial, rm)
    "only consider giant connected component? "
    Gcc = sorted(nx.connected_components(G), key=len, reverse=True)[0]
    survive = list(Gcc)
    A_update = A[survive, :][:, survive] 
    initial_update = initial[survive]
    return A_update, initial_update

def betaspace(A, x):
    """calculate  beta_eff and x_eff from A and x

    :A: adjacency matrix
    :x: state vector
    :returns: TODO

    """
    s_out = A.sum(0)
    s_in = A.sum(-1)
    if sum(s_out) == 0:
        return 0, x[0] 
    else:
        beta_eff = np.mean(s_out * s_in) / np.mean(s_out)
        if np.ndim(x) == 1:
            x_eff = np.mean(x * s_out)/ np.mean(s_out)
        elif np.ndim(x) == 2:  # x is matrix form 
            x_eff = np.mean(x * s_out, -1)/np.mean(s_out)
        return beta_eff, x_eff

def stable_state(A, A_interaction, index_i, index_j, cum_index, arguments, low=0.1, high=10, d=None):
    """calculate stables states for a given interaction matrix and dynamics-main.mutual

    :A: Adjacency matrix
    :degree: degree of lattice 
    :returns: stable states for all nodes x_l, x_h

    """
    dynamics = mutual_multi
    t = np.arange(0, 1000, 0.01)
    N = np.size(A, 0)
    xs_low = odeint(dynamics, np.ones(N) * low, t, args=(N, index_i, index_j, A_interaction, cum_index, arguments))[-1]
    xs_high = odeint(dynamics, np.ones(N) * high, t, args=(N, index_i, index_j, A_interaction, cum_index, arguments))[-1]
    return xs_low, xs_high

def normalization_x(x, xs_low, xs_high):
    """TODO: Docstring for normalization_x.

    :x: TODO
    :returns: TODO

    """
    R = (x-xs_low)/(xs_high-xs_low)
    return R

def mutual_1D(x, t, c, arguments):
    """original dynamics N species interaction.

    :x: N dynamic variables, 1 * N vector 
    :t: time 
    :N: the number of interacting variables 
    :index: the index of non-zero element of adjacency matrix A, N * N matrix 
    :neighbor: the degree of each node 
    :A_interaction: non-zero element of adjacency matrix A, 1 * N vector
    :returns: derivative of x 

    """
    B, C, D, E, H, K = arguments
    dxdt = B + x * (1 - x/K) * ( x/C - 1) + c * x**2 / (D + (E+H) * x)
    return dxdt

def mutual_multi(x, t, N, index_i, index_j, A_interaction, cum_index, arguments):
    """original dynamics N species interaction.

    :x: N dynamic variables, 1 * N vector 
    :t: time 
    :N: the number of interacting variables 
    :index: the index of non-zero element of adjacency matrix A, N * N matrix 
    :degree: the degree of lattice  
    :A_interaction: non-zero element of adjacency matrix A, 1 * N vector
    :returns: derivative of x 

    """
    B, C, D, E, H, K = arguments
    x[np.where(x<0)] = 0  # Negative x is forbidden
    sum_f = B + x * (1 - x/K) * ( x/C - 1)
    sum_g = A_interaction * x[index_j] / (D + E * x[index_i] + H * x[index_j])

    dxdt = sum_f + x * np.array([sum_g[i:j].sum() for i, j in zip(cum_index[:-1], cum_index[1:])])

    return dxdt

def network_generate(network_type, N, beta, seed, d=None):
    """TODO: Docstring for network_generate.

    :arg1: TODO
    :returns: TODO

    """
    if network_type == '2D':
        G = nx.grid_graph(dim=[int(np.sqrt(N)),int(np.sqrt(N))], periodic=True)
    elif network_type == 'RR':
        G = nx.random_regular_graph(d, N, seed)
    elif network_type == 'ER':
        #G = nx.fast_gnp_random_graph(N, d, seed)
        m = d
        G = nx.gnm_random_graph(N, m, seed)
    elif network_type == 'BA':
        m = d
        G = nx.barabasi_albert_graph(N, m, seed)
    elif network_type == 'real':
        A, M, N = load_data(seed)
        A = A_from_data(seed%2, M)
    if network_type != 'real':
        A = np.array(nx.adjacency_matrix(G).todense()) 
    else:
        if nx.is_connected(G) == False:
            print('more than one component')
            return None
    beta_eff, _ = betaspace(A, [0])
    weight = beta/ beta_eff
    A = A * weight
    A_index = np.where(A>0)
    A_interaction = A[A_index]
    index_i = A_index[0] 
    index_j = A_index[1] 
    degree = np.sum(A>0, 1)
    cum_index = np.hstack((0, np.cumsum(degree)))
    return A, A_interaction, index_i, index_j, cum_index


