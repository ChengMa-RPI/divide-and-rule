import os
os.environ['OPENBLAS_NUM_THREADS'] ='1'
import numpy as np 
import pandas as pd
import networkx as nx
from collections import Counter
import functools
import operator
import multiprocessing as mp
from collections import defaultdict
import time
#from mutual_framework import network_generate
from helper_function  import all_state, transition_rule, network_generate

def actual_simulation_network(neighbors, N_actual, interaction_number, data_point, initial_state, state_single, comm_seed, des):
    """TODO: Docstring for actual_simulation.

    :number_opinion: TODO
    :N: TODO
    :interaction_number: TODO
    :interval: TODO
    :seed: TODO
    :pA: TODO
    :p: TODO
    :returns: TODO

    """
    interval = int(interaction_number/data_point)
    state_single_evolution = np.zeros((data_point, len(state_single)))
    state = initial_state.copy()
    random_state = np.random.RandomState(comm_seed)
    speaker_list = random_state.choice(N_actual, size=interaction_number, replace=True)
    listener_prelist = random_state.randint(0, N_actual, size=interaction_number)

    select_list = random_state.random(interaction_number)  # used to select one from more than one outcomes
    for i in range(interaction_number):
        speaker = speaker_list[i]
        #listener = random_state.choice(list(G.neighbors(speaker)))
        #listener = neighbors[speaker][random_state.randint(len(neighbors[speaker]))]
        listener = neighbors[speaker][listener_prelist[i] % len(neighbors[speaker])]
        speaker_state = state[speaker]
        listener_state = state[listener]
        transition_state_all = transition_rule(speaker_state, listener_state)
        transition_state_length = len(transition_state_all)
        if transition_state_length == 1:
            select_state = transition_state_all[0]
        else:
            select_number = select_list[i]
            select_state = transition_state_all[int(select_number * transition_state_length)]
        state[speaker] = select_state[0]
        state[listener] = select_state[1]
        if not i % interval:
            state_counter = Counter(state)
            j = int(i // interval)
            for single, state_index in zip(state_single, range(len(state_single))):
                state_single_evolution[j, state_index] = state_counter[single]

    des_file = des + f'comm_seed={comm_seed}.csv'
    df_data = pd.DataFrame(np.hstack((np.arange(0, interaction_number, interval).reshape(data_point, 1), state_single_evolution)))
    df_data.to_csv(des_file, index=None, header=None)
    return None

def actual_simulation_complete(N, interaction_number, data_point, initial_state, state_single, comm_seed, des):
    """TODO: Docstring for actual_simulation.

    :number_opinion: TODO
    :N: TODO
    :interaction_number: TODO
    :interval: TODO
    :seed: TODO
    :pA: TODO
    :p: TODO
    :returns: TODO

    """
    interval = int(interaction_number/data_point)
    state_single_evolution = np.zeros((data_point, len(state_single)))
    state = initial_state.copy()
    random_state = np.random.RandomState(comm_seed)
    speaker_list = random_state.choice(N, size=interaction_number, replace=True)
    listener_list = random_state.randint(0, N, size=interaction_number)
    listener_speaker_index = np.where((listener_list - speaker_list) == 0)[0]
    listener_speaker_list = np.random.RandomState(comm_seed+100).randint(1, N, size = len(listener_speaker_index)) 
    listener_list[listener_speaker_index] = (listener_list[listener_speaker_index] + listener_speaker_list) % N
    select_list = random_state.random(interaction_number)  # used to select one from more than one outcomes
    for i in range(interaction_number):
        speaker = speaker_list[i]
        listener = listener_list[i]
        speaker_state = state[speaker]
        listener_state = state[listener]
        transition_state_all = transition_rule(speaker_state, listener_state)
        transition_state_length = len(transition_state_all)
        if transition_state_length == 1:
            select_state = transition_state_all[0]
        else:
            select_number = select_list[i]
            select_state = transition_state_all[int(select_number * transition_state_length)]
        state[speaker] = select_state[0]
        state[listener] = select_state[1]
        if not i % interval:
            state_counter = Counter(state)
            j = int(i // interval)
            for single, state_index in zip(state_single, range(len(state_single))):
                state_single_evolution[j, state_index] = state_counter[single]

    des_file = des + f'comm_seed={comm_seed}.csv'
    df_data = pd.DataFrame(np.hstack((np.arange(0, interaction_number, interval).reshape(data_point, 1), state_single_evolution)))
    df_data.to_csv(des_file, index=None, header=None)
    return None

def parallel_actual_simulation_network_multi_opinion(network_type, N, net_seed, d, interaction_number, data_point, number_opinion, comm_seed_list, pA, p, fluctuate_seed=0, sigma_p=0, sigma_pu=0):
    """TODO: Docstring for parallel_actual_simlation.

    :arg1: TODO
    :returns: TODO

    """
    minority = number_opinion - 1
    neighbors = defaultdict() 
    if network_type == 'complete':
        N_actual = N
        for i in range(N_actual):
            neighbors[i] = np.setdiff1d(np.arange(N_actual), i)
    else:
        A, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, 1, 0, net_seed, d)
        N_actual = len(A)
        G = nx.from_numpy_array(A)
        for i in G.nodes():
            neighbors[i] = list(G.neighbors(i))


    state_list = all_state(number_opinion)
    state_single = state_list[0: number_opinion]
    state_committed = state_list[number_opinion: 2*number_opinion]
    state_Atilde_committed = state_committed[1:] 
    state_Atilde_uncommitted = state_single[1:]

    seed_p = fluctuate_seed
    seed_pu = fluctuate_seed
    ncA = int(round(N_actual * pA))
    pu = (1-pA - p * minority) / minority
    "p, pu is constant if sigma_p and sigma_pu == 0, which is the current setup. NON_ZERO sigma_p or sigma_pu are designed for further exporations on unequal distribution of committed/uncommitted agents among the group A_tilde (the collection of opinions other than the big committed opinion A)"
    """
    if sigma_p == 0 and sigma_pu == 0:
        p_fluctuate = p * np.ones(minority)
        pu_fluctuate = pu * np.ones(minority)
    """
    if True:
        while True:
            p_fluctuate = np.random.RandomState(seed_p).normal(p, sigma_p*p, minority)
            p_fluctuate = p_fluctuate / p_fluctuate.sum() * p*minority
            nc_minority =  np.array(np.round(N_actual * p_fluctuate), int)

            if nc_minority.min() > 0 and nc_minority.sum() == round(N_actual * p * minority):
                break
            else:
                seed_p += 1

        while True:
            pu_fluctuate = np.random.RandomState(seed_pu).normal(pu, sigma_pu*pu, minority)
            pu_fluctuate = pu_fluctuate / pu_fluctuate.sum() * pu*minority
            if pu_fluctuate.min() > 0 and pu_fluctuate.max() <= pu * minority:
                break
            else:
                seed_pu += 1

    nc_minority =  np.array(np.round(N_actual * p_fluctuate), int)
    nuc_minority =  np.array(np.round(N_actual * pu_fluctuate), int)
    nuc_remaining = N_actual - ncA - nc_minority.sum() - nuc_minority.sum()
    if nuc_remaining // minority:
        nuc_minority += nuc_remaining // minority
        nuc_remaining = nuc_remaining % minority

    initial_state = ['a'] * ncA + functools.reduce(operator.iconcat, [[i] * nc for i, nc in zip(state_Atilde_committed, nc_minority)], []) + functools.reduce(operator.iconcat, [[i] * nuc for i, nuc in zip(state_Atilde_uncommitted, nuc_minority)], []) + [i for i in state_Atilde_uncommitted[:nuc_remaining]]

    if fluctuate_seed==0 and sigma_p==0 and sigma_pu==0:
        des = f'../data/' + network_type + f'/N={N}_d={d}_netseed={net_seed}/actual_simulation/number_opinion={number_opinion}/interaction_number={interaction_number}_pA={pA}_p={p}/'
    else:
        des = f'../data/' + network_type + f'/N={N}_d={d}_netseed={net_seed}/actual_simulation_fluctuate/fluctuate_seed={fluctuate_seed}_sigma_p={sigma_p}_sigma_pu={sigma_pu}/number_opinion={number_opinion}/interaction_number={interaction_number}_pA={pA}_p={p}/'
    if not os.path.exists(des):
        os.makedirs(des)

    p = mp.Pool(cpu_number)
    if network_type == 'complete':
        p.starmap_async(actual_simulation_complete, [(N, interaction_number, data_point, initial_state, state_single, comm_seed, des) for comm_seed in comm_seed_list]).get()
    else:
        p.starmap_async(actual_simulation_network, [( neighbors, N_actual, interaction_number, data_point, initial_state, state_single, comm_seed, des) for comm_seed in comm_seed_list]).get()
    p.close()
    p.join()
    return None


"""
Parameters explained
network_type, N, d, net_seed are used in the function network_generate to generate networks.
network_type: complete--complete graph, ER--ER network, SF--scale free network
N: the number of nodes/agents in the network to exchange opinions
d: the parameter to control the network generation, 
0 for complete graph (no meaning), 
an integer for ER network, determining the edge density, k=d * 2 / N, 
a list for Scale-free network, [gamma, kmax, kmin], gamma is the scale free exponent paramter, kmax=0 means no restriction on the maximum of degree
net_seed: random seed to generate graph

cpu_number: the number of CPUs used for parallel computing
interaction_number: the number of interactions between agents to exchange opinions, usually a very large number to make sure that the system reaches the equilibrium
data_point: the number of data points to save to the predefined destination instead of all interaction process, much less than the interaction number
number_opinion: the number of single opinions, "m"
comm_seed: the random seed to select agents for exchanging opinions
pA: the fraction of committed agents supporting opinion A
p: the fraction of committed agents supporting minority B, C, D, ...
number_opinion: the number of single opinions, "m"
comm_seed: the random seed to select agents for exchanging opinions
pA: the fraction of committed agents supporting opinion A
p: the fraction of committed agents supporting minority B, C, D, ...
"""

cpu_number = 8
N = 1000
interaction_number = 1000 * 1000

"complete graph"
network_type = 'complete'
net_seed_list = [0]
d_list = [0]

"Scale-free network of different power-law exponent"
network_type = 'SF'
d_list = [[2.1, 0, 2], [2.5, 0, 3], [3.5, 0, 4]]
net_seed_list = [[5, 5], [98, 98], [79, 79]]  # networks of these selected random seeds have the connected component with nodes close to N

d_list = [[3.8, 0, 5]]
net_seed_list = [[0, 0]]  # networks of these selected random seeds have the connected component with nodes close to N

"Watts_strogatz network"
network_type = 'WS'
d_list = [[6, 0.2]]
net_seed_list = [0]

"ER networks of different edge densities"
network_type = 'ER'
net_seed_list = [1, 0, 0, 0]
d_list = [3000, 4000, 8000, 16000]

data_point = 1000
number_opinion = 6
p = 0.012
comm_seed_list = np.arange(10, 50, 1)  # for averaging effect, multiple communication seeds to generate random realizations of naming game interaction
pA_list = [0.01]
pA_list = np.round(np.arange(0.1, 0.16, 0.01), 3)
for d, net_seed in zip(d_list, net_seed_list):
    for pA in pA_list:
        parallel_actual_simulation_network_multi_opinion(network_type, N, net_seed, d, interaction_number, data_point, number_opinion, comm_seed_list, pA, p)
        pass
