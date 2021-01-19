import numpy as np
import Utils
import math
import random
import theo
from sklearn.metrics import r2_score
from networkx.readwrite import json_graph
import json
import networkx as nx

with open('er_2000_20.json', 'r') as f:
    G = json_graph.node_link_graph(json.load(f))

T = 40
iter = 1
data_index = 6
node_num = 2000
adj = np.array(nx.adjacency_matrix(G).todense())


def simu_single(vac_ava1, vac_ava2, T, beta1_U, beta1_P, beta2_U, beta2_P, gamma1, gamma2):
    health_states1 = np.zeros(node_num)
    health_states2 = np.zeros(node_num)

    vac_nodes1 = random.sample(list(range(node_num)), vac_ava1)
    health_states1[vac_nodes1] = 2
    vac_nodes2 = random.sample(list(range(node_num)), vac_ava2)
    health_states2[vac_nodes2] = 2

    init_infected_candidate1 = list(np.where(health_states1 == 0)[0])
    if len(init_infected_candidate1) > 0:
        init_infected_node1 = random.choice(init_infected_candidate1)
        health_states1[init_infected_node1] = 1

    init_infected_candidate2 = list(np.where(health_states2 == 0)[0])
    if len(init_infected_candidate2) > 0:
        init_infected_node2 = random.choice(init_infected_candidate2)
        health_states2[init_infected_node2] = 1

    S1_all = [len(np.where(health_states1 == 0)[0]) / node_num]
    I1_all = [len(np.where(health_states1 == 1)[0]) / node_num]
    R1_all = [len(np.where(health_states1 == 2)[0]) / node_num]

    S2_all = [len(np.where(health_states2 == 0)[0]) / node_num]
    I2_all = [len(np.where(health_states2 == 1)[0]) / node_num]
    R2_all = [len(np.where(health_states2 == 2)[0]) / node_num]

    for t in range(T):
        susceptibility_states = Utils.get_sus_in_vac_simu_1(beta1_U, beta1_P, beta2_U, beta2_P, health_states1,
                                                            health_states2)
        health_states1 = Utils.health_state_trans(adj, [gamma1, gamma2], health_states1, susceptibility_states, 1)

        health_states2 = Utils.health_state_trans(adj, [gamma1, gamma2], health_states2, susceptibility_states, 2)

        S1_all.append(len(np.where(health_states1 == 0)[0]) / node_num)
        I1_all.append(len(np.where(health_states1 == 1)[0]) / node_num)
        R1_all.append(len(np.where(health_states1 == 2)[0]) / node_num)

        S2_all.append(len(np.where(health_states2 == 0)[0]) / node_num)
        I2_all.append(len(np.where(health_states2 == 1)[0]) / node_num)
        R2_all.append(len(np.where(health_states2 == 2)[0]) / node_num)

    if len(S1_all) < T:
        S1_all.extend([S1_all[-1]] * (T + 1 - len(S1_all)))
        I1_all.extend([I1_all[-1]] * (T + 1 - len(I1_all)))
        R1_all.extend([R1_all[-1]] * (T + 1 - len(R1_all)))
        S2_all.extend([S2_all[-1]] * (T + 1 - len(S2_all)))
        I2_all.extend([I2_all[-1]] * (T + 1 - len(I2_all)))
        R2_all.extend([R2_all[-1]] * (T + 1 - len(R2_all)))

    return S1_all, I1_all, R1_all, S2_all, I2_all, R2_all


def simu_ave(iter, vac_ava1, vac_ava2, beta1_U, beta1_P, beta2_U, beta2_P, gamma1, gamma2):
    S1_ALL = np.zeros(T + 1)
    R1_ALL = np.zeros(T + 1)
    S2_ALL = np.zeros(T + 1)
    R2_ALL = np.zeros(T + 1)

    for i in range(iter):
        S1_all, I1_all, R1_all, S2_all, I2_all, R2_all = simu_single(vac_ava1, vac_ava2, T, beta1_U, beta1_P, beta2_U,
                                                                     beta2_P,
                                                                     gamma1, gamma2)

        S1_ALL += np.array(S1_all)
        R1_ALL += np.array(R1_all)
        S2_ALL += np.array(S2_all)
        R2_ALL += np.array(R2_all)

    S1_ALL = S1_ALL / iter
    R1_ALL = R1_ALL / iter
    S2_ALL = S2_ALL / iter
    R2_ALL = R2_ALL / iter

    S1, I1, R1, S2, I2, R2 = theo.theoretical(vac_ava1, vac_ava2, T, beta1_U, beta1_P, beta2_U, beta2_P,
                                              gamma1, gamma2, adj, node_num)

    return S1_ALL[-1] + R1_ALL[0], S2_ALL[-1] + R2_ALL[0], S1[-1] + R1[0], S2[-1] + R2[0]


alpha1 = 0.1
beta1_U = 0.4
beta1_P = alpha1 * beta1_U
gamma1 = 0.8

alpha2 = 0.1
beta2_U = 0.5
beta2_P = alpha2 * beta2_U
gamma2 = 0.6


vac_ava1 = math.floor(0.5 * node_num)
vac_ava2 = math.floor(node_num * 0.5)


S1_coff = []
s1_coff = []
S2_coff = []
s2_coff = []


for coff in np.arange(0, 1.01, 0.02):
    vac_ava1 = math.floor(node_num * coff)
    S1, S2, s1, s2 = simu_ave(iter, vac_ava1, vac_ava2, beta1_U, beta1_P, beta2_U, beta2_P, gamma1, gamma2)
    S1_coff.append(S1)
    s1_coff.append(s1)
    S2_coff.append(S2)
    s2_coff.append(s2)


r2_1 = r2_score(S1_coff, s1_coff)
r2_2 = r2_score(S2_coff, s2_coff)