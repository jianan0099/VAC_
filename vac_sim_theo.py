# theoretical analysis markov 方式
import networkx as nx
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import Utils

def init_P1(vac_ava1, vac_ava2, node_num):
    # P node_num * 9 初始 每层只有一个I 打第一种疫苗人数为 vac_ava1 打第二种疫苗人数为 vac_ava2
    P = np.zeros((node_num, 9))
    if vac_ava1 < node_num:
        x = np.array([1 - 1 / node_num - vac_ava1 / node_num, 1 / node_num, vac_ava1 / node_num])
    else:
        x = np.array([0, 0, 1])

    if vac_ava2 < node_num:
        y = np.array([1 - 1 / node_num - vac_ava2 / node_num, 1 / node_num, vac_ava2 / node_num])
    else:
        y = np.array([0, 0, 1])
    init_prob = np.array([i * y for i in x]).flatten()
    for i in range(9):
        P[:, i] = init_prob[i]
    return P


def get_p_I(l, P):
    """
    RETURN array node_num * 1 每个点在 L 层 为 I 的概率
    """
    if l == 1:
        q_I = P[:, 3] + P[:, 4] + P[:, 5]

    if l == 2:
        q_I = P[:, 1] + P[:, 4] + P[:, 7]
    return q_I


def get_q_l(l, P, U_or_P, beta1_U, beta1_P, beta2_U, beta2_P, adj, node_num):
    # 计算 q l # 矩阵化 node_num * 1 第l层
    if U_or_P == 0:
        # U
        beta = beta1_U if 1 == l else beta2_U
    else:
        beta = beta1_P if 1 == l else beta2_P

    q_I = get_p_I(l, P)
    p_l_temp = 1 - np.multiply(adj, np.tile(q_I, (node_num, 1))) * beta

    p_l = np.prod(p_l_temp, axis=1)  # 每行积
    return p_l


def theoretical(vac_ava1, vac_ava2, T, beta1_U, beta1_P, beta2_U, beta2_P, gamma1, gamma2, adj, node_num):
    P = init_P1(vac_ava1, vac_ava2, node_num)
    S1 = []
    I1 = []
    R1 = []
    S2 = []
    I2 = []
    R2 = []

    S1.append(sum(P[:, 0] + P[:, 1] + P[:, 2]) / node_num)
    I1.append(sum(P[:, 3] + P[:, 4] + P[:, 5]) / node_num)
    R1.append(sum(P[:, 6] + P[:, 7] + P[:, 8]) / node_num)
    S2.append(sum(P[:, 0] + P[:, 3] + P[:, 6]) / node_num)
    I2.append(sum(P[:, 1] + P[:, 4] + P[:, 7]) / node_num)
    R2.append(sum(P[:, 2] + P[:, 5] + P[:, 8]) / node_num)
    for t in range(T):
        # 循环一下过程
        q1U = get_q_l(1, P, 0, beta1_U, beta1_P, beta2_U, beta2_P, adj, node_num)
        q1P = get_q_l(1, P, 1, beta1_U, beta1_P, beta2_U, beta2_P, adj, node_num)
        q2U = get_q_l(2, P, 0, beta1_U, beta1_P, beta2_U, beta2_P, adj, node_num)
        q2P = get_q_l(2, P, 1, beta1_U, beta1_P, beta2_U, beta2_P, adj, node_num)

        SS = np.multiply(np.multiply(q1U, q2U), P[:, 0])
        SI = np.multiply(np.multiply(q1U, 1 - q2U), P[:, 0]) + np.multiply(q1U * (1 - gamma2), P[:, 1])
        SR = np.multiply(q1U * gamma2, P[:, 1]) + np.multiply(q1P, P[:, 2])
        IS = np.multiply(np.multiply(q2U, 1 - q1U), P[:, 0]) + np.multiply(q2U * (1 - gamma1), P[:, 3])
        II = np.multiply(np.multiply(1 - q1U, 1 - q2U), P[:, 0]) + np.multiply((1 - q1U) * (1 - gamma2), P[:, 1]) \
             + np.multiply((1 - q2U) * (1 - gamma1), P[:, 3]) + (1 - gamma1) * (1 - gamma2) * P[:, 4]
        IR = gamma2 * (1 - gamma1) * P[:, 4] + (1 - gamma1) * P[:, 5] + np.multiply((1 - q1U) * gamma2, P[:, 1]) \
             + np.multiply((1 - q1P), P[:, 2])
        RS = np.multiply(q2U * gamma1, P[:, 3]) + np.multiply(q2P, P[:, 6])
        RI = np.multiply(gamma1 * (1 - q2U), P[:, 3]) + np.multiply(1 - q2P, P[:, 6]) + gamma1 * (1 - gamma2) * P[:, 4] \
             + (1 - gamma2) * P[:, 7]
        RR = gamma1 * gamma2 * P[:, 4] + gamma1 * P[:, 5] + gamma2 * P[:, 7] + P[:, 8]

        # 更新 P
        P[:, 0] = SS
        P[:, 1] = SI
        P[:, 2] = SR
        P[:, 3] = IS
        P[:, 4] = II
        P[:, 5] = IR
        P[:, 6] = RS
        P[:, 7] = RI
        P[:, 8] = RR

        S1.append(sum(P[:, 0] + P[:, 1] + P[:, 2]) / node_num)
        I1.append(sum(P[:, 3] + P[:, 4] + P[:, 5]) / node_num)
        R1.append(sum(P[:, 6] + P[:, 7] + P[:, 8]) / node_num)
        S2.append(sum(P[:, 0] + P[:, 3] + P[:, 6]) / node_num)
        I2.append(sum(P[:, 1] + P[:, 4] + P[:, 7]) / node_num)
        R2.append(sum(P[:, 2] + P[:, 5] + P[:, 8]) / node_num)

    return S1, I1, R1, S2, I2, R2


