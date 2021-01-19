import numpy as np
from scipy.stats import bernoulli
import networkx as nx
import json
from networkx.readwrite import json_graph

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
                                                 size=sum_of_possible_trans_link[possible_trans_S[i]])  # p为取1的概率
            if sum(random_choice_result) > 0:
                infection_result.append(1)
            else:
                infection_result.append(0)
        health_state[S_nodes[possible_trans_S]] = infection_result

    if len(I_nodes) != 0:
        recover_result = bernoulli.rvs(gamma_all[disease_index - 1], size=len(I_nodes)) + 1
        health_state[I_nodes] = recover_result
    return health_state

def all_net_info(index):
    """
    index 表示选择哪个网络
    """
    # ----- 所有网络信息 -------------------------
    G_all = {1: read_saved_G('er_2000_5.json'),
             2: read_saved_G('er_2000_15.json'),
             3: read_saved_G('er_5000_5.json'),
             4: read_saved_G('er_5000_10.json'),
             5: read_saved_G('er_20000_50.json'),
             6: read_saved_G('er_2000_20.json'),
             7: read_saved_G('er_2000_2.json'),
             8: read_saved_G('er_2000_10.json')}
    # adj = np.array(nx.adjacency_matrix(G).todense())

    degree_info_all = {1: get_degree_info_from_json('degree_info_2000_5.json'),
                       2: get_degree_info_from_json('degree_info_2000_15.json'),
                       3: get_degree_info_from_json('degree_info_5000_5.json'),
                       4: get_degree_info_from_json('degree_info_5000_10.json'),
                       5: get_degree_info_from_json('degree_info_20000_50.json'),
                       6: get_degree_info_from_json('degree_info_2000_20.json'),
                       7: get_degree_info_from_json('degree_info_2000_2.json'),
                       8: get_degree_info_from_json('degree_info_2000_10.json')}
    # degree_list, distribution_list = Utils.get_degree_info_from_json('degree_info_5000_5.json')

    node_num_all = {1: 2000,
                    2: 2000,
                    3: 5000,
                    4: 5000,
                    5: 20000,
                    6: 2000,
                    7: 2000,
                    8: 2000}
    # -------------------------------------------
    G = G_all[index]
    adj = np.array(nx.adjacency_matrix(G).todense())
    degree_list, distribution_list = degree_info_all[index]
    node_num = node_num_all[index]
    return G, adj, degree_list, distribution_list, node_num


def get_degree_info_from_json(path):
    """
    打开path中的json文件 获取对应的degree distribution信息
    返回的两个都是List
    """
    with open(path, 'r') as load_f:
        degree_dict = json.load(load_f)
    degree_list = degree_dict['degree']
    distribution_list = degree_dict['distribution']
    return degree_list, distribution_list

def read_saved_G(path):
    """
    获取之前保存的图
    """
    with open(path, 'r') as f:
        G = json_graph.node_link_graph(json.load(f))
    return G




