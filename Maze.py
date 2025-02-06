import numpy as np
import tkinter as tk
import time
import random
import math
from scipy.spatial.distance import cdist
import scipy.io

from copy import deepcopy


np.random.seed(1)

# L_X = 8
# L_Y = 8
# L_Z = 3
# L_H = 3

# L_X = 8
# L_Y = 3

L_X = 5

p = 0.1
obs_thres=[[0,1000],[5,15]]


d_ite = 100

# use_num = 10
# use_num = 10

UAV_fly = 300

# UAV_fly = 186

t12 = 20
# t12 = 2

pmn = 0.1

p_j = 0.040
# p_j = 0.025

# b_w = 1000000
# b_w1 = 100000

noise_power = 7e-11

k_n = 2

def channel_gain(uav_center00, uav_center01, userte_center10, userte_center11, UAV_h):
    # 视距信道概率
    # 用户与无人机间的距离
    dis_us2u2 = np.sqrt((uav_center00 - userte_center10)
                        ** 2 + (uav_center01 - userte_center11) ** 2)  # 用户与无人机间的二维距离
    dis_us2u3 = np.sqrt((uav_center00 - userte_center10)
                        ** 2 + (uav_center01 - userte_center11) ** 2 + UAV_h ** 2)  # 用户与无人机间的三维距离

    # print('dis_us2u2: ', dis_us2u2)
    # print('dis_us2u3: ', dis_us2u3)

    # 信道增益计算
    # 用户与无人机间的信道增益

    theta1 = (180 / math.pi) * np.arctan(UAV_h / dis_us2u2)
    # print('theta1[i][j]:', theta1[i][j])
    plos = 1 / (1 + 10.72 * np.exp(-0.05 * (theta1 - 10.72)))
    gus2u = 10 ** - ((((0.1 - 21)) * plos + 21) / 10) / ((dis_us2u3) ** 2 * (4 * math.pi / 3) ** 2)

    # print('gus2u: ', gus2u)
    return gus2u


class Maze(object):
    def __init__(self):
        super(Maze, self).__init__()

        self.n_actions = 38880
        # self.n_actions = 68040
        self.n_features = 9  # 无人机位置, 电池量，信道接入情况
        # self._build_maze()

    # def step(self, action, uav_center2, step1, user_center2, rit, jam_center2,
    #          uav_center3, user_center3, jam_center3, userte_center2, episodes):
    def step(action, uav_center2, step1, user_center2, rit, jam_center2,
             uav_center3, user_center3, userte_center2, episodes):

        # print('step1: ', step1)
        # print('action: ', action)
        # print('action_1: ', action_1)

        action1 = step1 * 2
        action2 = step1 * 2 + 1

        action1_dex = action1
        action2_dex = action2


        action_1_1 = round((action[4] + 1) / 2 * 1)
        action_1_2 = round((action[5] + 1) / 2 * 4)
        action_1_3 = round((action[6] + 1) / 2 * 1)
        action_1_4 = round((action[7] + 1) / 2 * 1)
        action_1_5 = round((action[8] + 1) / 2 * 4)
        action_1_6 = round((action[9] + 1) / 2 * 1)






        if action_1_3 == 0:
            action1 = action1
            action2 = action2
        else:
            action1 = action2_dex
            action2 = action1_dex



        action3 = action_1_2
        action4 = action_1_5


        # print('action1: ', action1)
        # print('action2: ', action2)

        ########## 干扰机向中心点方向移动

        # 定义目标点
        target_point = np.array([500, 500])

        # 计算移动方向的单位向量
        direction_vector = target_point - jam_center2
        unit_direction_vector = direction_vector / np.linalg.norm(direction_vector)

        # 定义移动距离
        distance = 14  # 10 米

        # 计算移动向量
        move_vector = unit_direction_vector * distance

        # 移动点
        jam_center2 = jam_center2 + move_vector

        ###################################




        ###

        uav_center2[0][0] = uav_center2[0][0] + UAV_fly * action[0] * math.cos(2 * math.pi * action[1])
        uav_center2[0][1] = uav_center2[0][1] + UAV_fly * action[0] * math.sin(2 * math.pi * action[1])
        uav_center2[1][0] = uav_center2[1][0] + UAV_fly * action[2] * math.cos(2 * math.pi * action[3])
        uav_center2[1][1] = uav_center2[1][1] + UAV_fly * action[2] * math.sin(2 * math.pi * action[3])


        disuav1_2 = math.sqrt(
            (uav_center2[0][0] - uav_center2[1][0]) ** 2 + (uav_center2[0][1] - uav_center2[1][1]) ** 2)

        reward_7 = 0
        if disuav1_2 < 5:
            reward_7 = - 0.1


        vt = 20

        #
        disyd1 = math.sqrt((UAV_fly * action[0] * math.cos(2 * math.pi * action[1])) ** 2 + (
                    UAV_fly * action[0] * math.sin(2 * math.pi * action[1])) ** 2)
        disyd2 = math.sqrt((UAV_fly * action[2] * math.cos(2 * math.pi * action[3])) ** 2 + (
                    UAV_fly * action[2] * math.sin(2 * math.pi * action[3])) ** 2)

        # print('disyd1: ', disyd1)
        # print('disyd2: ', disyd2)

        t_yd1 = disyd1 / vt  #
        t_yd2 = disyd2 / vt  #



        g_d_1 = channel_gain(uav_center2[0][0], uav_center2[0][1],
                             userte_center2[action3][0], userte_center2[action3][1], 100)


        g_d_2 = channel_gain(uav_center2[1][0], uav_center2[1][1],
                             userte_center2[action4][0], userte_center2[action4][1], 100)


        g_d_1u = channel_gain(uav_center2[0][0], uav_center2[0][1],
                             user_center2[action1][0], user_center2[action1][1], 100)


        g_d_1uj = channel_gain(uav_center2[0][0], uav_center2[0][1],
                             jam_center2[0][0], jam_center2[0][1], 100)


        g_d_2u = channel_gain(uav_center2[1][0], uav_center2[1][1],
                             user_center2[action2][0], user_center2[action2][1], 100)


        g_d_2uj = channel_gain(uav_center2[1][0], uav_center2[1][1],
                             jam_center2[0][0], jam_center2[0][1], 100)


        ############ SNR

        #####
        s_u1t = pmn * g_d_1 / (p_j * g_d_1uj + noise_power)
        s_u2t = pmn * g_d_2 / (p_j * g_d_2uj + noise_power)

        #####
        s_u1u = pmn * g_d_1u / (p_j * g_d_1uj + noise_power)
        s_u2u = pmn * g_d_2u / (p_j * g_d_2uj + noise_power)



        s_u1t = np.round(10 * math.log10(s_u1t))
        s_u2t = np.round(10 * math.log10(s_u2t))
        s_u1u = np.round(10 * math.log10(s_u1u))
        s_u2u = np.round(10 * math.log10(s_u2u))


        s_u1t = np.clip(s_u1t, -10, 20) + 10
        s_u2t = np.clip(s_u2t, -10, 20) + 10
        s_u1u = np.clip(s_u1u, -10, 20) + 10
        s_u2u = np.clip(s_u2u, -10, 20) + 10


        # 从 MATLAB 文件加载数据
        mat_data = scipy.io.loadmat('fit21.mat')

        semdm_table = mat_data['fit21']


        #####
        varpi1 = semdm_table[int(s_u1t)][int(s_u1u)]
        varpi2 = semdm_table[int(s_u2t)][int(s_u2u)]



        ##########################################################

        ##### 第一个无人机和第二个无人机的语义速率

        if action_1_1 >= 0:
            bandkt_1 = 1000000/2
            bandkt_2 = 1560000/2
            banket_ch1 = 15
            banket_ch2 = 25
        else:
            bandkt_1 = 1560000/2
            bandkt_2 = 1000000/2
            banket_ch1 = 25
            banket_ch2 = 15

        bandkt_te = 50000


        r_mn1 = bandkt_1 * ((4 * 197) / (k_n * 197)) * varpi1
        r_mnte1 = bandkt_te * (2 / 4) * varpi1

        r_mn2 = bandkt_2 * ((4 * 197) / (k_n * 197)) * varpi2
        r_mnte2 = bandkt_te * (2 / 4) * varpi2

        reward1_4 = 0
        reward1_5 = 0


        d_ite_1 = 5000


        d_i1 = rit[action1]
        rit[action1] = rit[action1] - d_i1
        # print('d_i1:', d_i1)
        # if rit[action1] > 0:
        #     reward1_4 = 0.1 * rit[action1]

        t_com_1_uav1 = (d_i1*40*197) / (r_mn1)

        t_com_2_uav1 = (d_i1) / (1.4) + 0.01

        # print('t_com_2_uav1:', t_com_2_uav1)

        t_comte_1_uav1 = (d_ite_1*40) / (r_mnte1)

        t_comte_2_uav1 = (d_ite_1 * 1.2) / (1.4 * 10 ** 3)


        d_ite_2 = 5000

        d_i2 = rit[action2] #
        rit[action2] = rit[action2] - d_i2


        t_com_1_uav2 = (d_i2*40*197) / (r_mn2)
        t_com_2_uav2 = (d_i2) / (1.5) + 0.01



        t_comte_1_uav2 = (d_ite_2*40) / (r_mnte2)

        t_comte_2_uav2 = (d_ite_2 * 1.2) / (1.5 * 10 ** 3)

        tcompu_sum1 = t_com_2_uav1 + t_comte_2_uav1
        tcompu_sum2 = t_com_2_uav2 + t_comte_2_uav2



        tsum = t_com_1_uav1 + t_comte_1_uav1 + t_com_1_uav2 + t_comte_1_uav2 \
               + t_com_2_uav1 + t_comte_2_uav1 + t_com_2_uav2 + t_comte_2_uav2

        tsum_com = t_com_1_uav1 + t_comte_1_uav1 + t_com_1_uav2 + t_comte_1_uav2

        tsum_compu = t_com_2_uav1 + t_comte_2_uav1 + t_com_2_uav2 + t_comte_2_uav2


        r_mn1_nsc = bandkt_1 * math.log2(1 + pmn * g_d_1u / (p_j * g_d_1uj + noise_power))
        r_mnte1_nsc = bandkt_te * math.log2(1 + pmn * g_d_1 / (p_j * g_d_1uj + noise_power))

        t_com_1_uav1_nsc = (d_i1 * 1000000) / r_mn1_nsc
        t_comte_1_uav1_nsc = (d_ite * 1200) / r_mnte1_nsc

        r_mn2_nsc = bandkt_2 * math.log2(1 + pmn * g_d_2u / (p_j * g_d_2uj + noise_power))
        r_mnte2_nsc = bandkt_te * math.log2(1 + pmn * g_d_2 / (p_j * g_d_2uj + noise_power))

        t_com_1_uav2_nsc = (d_i2 * 1000000) / r_mn2_nsc
        t_comte_1_uav2_nsc = (d_ite * 1200) / r_mnte2_nsc



        tsum_nsc = t_com_1_uav1_nsc + t_comte_1_uav1_nsc + t_com_1_uav2_nsc + t_comte_1_uav2_nsc \
                   + t_com_2_uav1 + t_comte_2_uav1 + t_com_2_uav2 + t_comte_2_uav2

        tsum_com_nsc = t_com_1_uav1_nsc + t_comte_1_uav1_nsc + t_com_1_uav2_nsc + t_comte_1_uav2_nsc



        ############################

        if uav_center2[0][0] > 1000:
            uav_center2[0][0] = 1000
        if uav_center2[0][0] < 0:
            uav_center2[0][0] = 0
        if uav_center2[0][1] > 1000:
            uav_center2[0][1] = 1000
        if uav_center2[0][1] < 0:
            uav_center2[0][1] = 0
        if uav_center2[1][0] > 1000:
            uav_center2[1][0] = 1000
        if uav_center2[1][0] < 0:
            uav_center2[1][0] = 0
        if uav_center2[1][1] > 1000:
            uav_center2[1][1] = 1000
        if uav_center2[1][1] < 0:
            uav_center2[1][1] = 0

        reward1_2 = 0
        reward1_3 = 0


        reward1_2 = 0
        reward1_3 = 0


        reward1_2 = varpi1 - 0.8
        reward1_3 = varpi2 - 0.8

        reward1_4 = 0
        reward1_5 = 0

        if varpi1 < 0.8:
            reward1_4 = 0.1

        if varpi2 < 0.8:
            reward1_5 = 0.1




        t_com_1_uav1 = t_com_1_uav1 + t_comte_1_uav1
        t_com_1_uav2 = t_com_1_uav2 + t_comte_1_uav2



        reward_uav1 = - 0.1 * 0.012 * (t_com_1_uav1 + tcompu_sum1)\
                      + 0.8 * reward1_2 + reward_7
        reward_uav2 = - 0.1 * 0.012 * (t_com_1_uav2 + tcompu_sum2)\
                      + 0.8 * reward1_3 + reward_7


        varpi_sum = varpi1 + varpi2
        rmn_sum = r_mn1 + r_mnte1 + r_mn2 + r_mnte2


        reward = reward_uav1 + reward_uav2

        if step1 == 5:
            done = True
        else:
            done = False



        user_center2[0][0] = user_center2[0][0] + 1
        user_center2[0][1] = user_center2[0][1] + 1



        s_ = np.hstack((uav_center2[0][0], uav_center2[0][1], 100, uav_center2[1][0], uav_center2[1][1], 100,
                        jam_center2[0][0], jam_center2[0][1], 760, 380, 270, 160, 500, 500, 570, 120, 470, 600,
                        user_center2[0][0], user_center2[0][1], user_center2[1][0], user_center2[1][1],
                        user_center2[2][0], user_center2[2][1], user_center2[3][0], user_center2[3][1],
                        user_center2[4][0], user_center2[4][1], user_center2[5][0], user_center2[5][1],
                        user_center2[6][0], user_center2[6][1], user_center2[7][0], user_center2[7][1],
                        user_center2[8][0], user_center2[8][1], user_center2[9][0], user_center2[9][1],
                        user_center2[10][0], user_center2[10][1], user_center2[11][0], user_center2[11][1], 0,
                        banket_ch1, 20 * varpi1, banket_ch2, 20 * varpi2))

        return s_, reward, done, uav_center2, user_center2, rit, jam_center2,\
            uav_center3, user_center3, varpi_sum, tsum, tsum_nsc







