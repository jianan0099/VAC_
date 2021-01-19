import numpy as np
from scipy.stats import bernoulli


def get_sus_in_vac_simu_1(beta1_U, beta1_P, beta2_U, beta2_P, health_states1, health_states2):
    sus = np.zeros((len(health_states1), 2))
    temp = health_states2 - 1
    sus[np.where(health_states1 == 0)[0], 0] = beta1_U
    sus[np.where((temp > 0) & (health_states1 == 0))[0], 0] = beta1_P
    temp = health_states1 - 1
    sus[np.where(health_states2 == 0)[0], 1] = beta2_U
    sus[np.where((temp > 0) & (health_states2 == 0))[0], 1] = beta2_P
    return sus


def health_state_trans(adj, gamma_all, health_state, susceptibility_state, disease_index):
    trans_disease = np.array(health_state == 1, dtype=np.int)
    S_nodes = np.where(health_state == 0)[0]
    I_nodes = np.where(health_state == 1)[0]

    if len(S_nodes) != 0:

        adj_S = adj[S_nodes]
        sus_S = susceptibility_state[S_nodes, disease_index - 1]
        multi_with_adj_S = np.tile(trans_disease, (len(adj_S), 1))
        possible_trans_link = np.multiply(adj_S, multi_with_adj_S)
        sum_of_possible_trans_link = np.sum(possible_trans_link, axis=1)
        infection_result = []
        possible_trans_S = np.nonzero(sum_of_possible_trans_link)[0]

        for i in range(len(possible_trans_S)):
            random_choice_result = bernoulli.rvs(sus_S[possible_trans_S[i]],
                                                 size=sum_of_possible_trans_link[possible_trans_S[i]])
            if sum(random_choice_result) > 0:
                infection_result.append(1)
            else:
                infection_result.append(0)
        health_state[S_nodes[possible_trans_S]] = infection_result

    if len(I_nodes) != 0:
        recover_result = bernoulli.rvs(gamma_all[disease_index - 1], size=len(I_nodes)) + 1
        health_state[I_nodes] = recover_result
    return health_state





