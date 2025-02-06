import argparse
from itertools import count

import os, sys, random
import numpy as np

# import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
# from tensorboardX import SummaryWriter
from Maze import Maze

import matplotlib.pyplot as plt
import numpy as np
import time
# import pyttsx3
from scipy.io import savemat
import random
import torch
from Maze import Maze


start = time.time()

'''
Implementation of Deep Deterministic Policy Gradients (DDPG) with pytorch 
riginal paper: https://arxiv.org/abs/1509.02971
Not the author's implementation !
'''

parser = argparse.ArgumentParser()
# parser.add_argument('--mode', default='train', type=str) # mode = 'train' or 'test'
# OpenAI gym environment name, # ['BipedalWalker-v2', 'Pendulum-v0'] or any continuous environment
# Note that DDPG is feasible about hyper-parameters.
# You should fine-tuning if you change to another environment.
# parser.add_argument("--env_name", default="Pendulum-v0")
parser.add_argument("--env_name", default="Maze")
# parser.add_argument('--tau',  default=0.1, type=float) # target smoothing coefficient
parser.add_argument('--tau',  default=0.1, type=float) # target smoothing coefficient
parser.add_argument('--target_update_interval', default=1, type=int)
# parser.add_argument('--test_iteration', default=10, type=int)

# parser.add_argument('--learning_rate', default=1e-4, type=float)
# parser.add_argument('--gamma', default=0.90, type=int) # discounted factor
parser.add_argument('--gamma', default=0.90, type=int) # discounted factor
# parser.add_argument('--capacity', default=3000, type=int) # replay buffer size
parser.add_argument('--capacity', default=2200, type=int) # replay buffer size
parser.add_argument('--batch_size', default=300, type=int) # mini batch size
parser.add_argument('--seed', default=True, type=bool)
# parser.add_argument('--random_seed', default=323, type=int)
parser.add_argument('--random_seed', default=9527, type=int)
# optional parameters

parser.add_argument('--sample_frequency', default=100, type=int)
parser.add_argument('--render', default=False, type=bool) # show UI or not
# parser.add_argument('--log_interval', default=50, type=int) #
parser.add_argument('--load', default=False, type=bool) # load model
# parser.add_argument('--render_interval', default=100, type=int) # after render_interval, the env.render() will work
# parser.add_argument('--exploration_noise', default=0.4, type=float)
parser.add_argument('--exploration_noise', default=0.3, type=float)
# parser.add_argument('--max_episode', default=1000, type=int) # num of games
# parser.add_argument('--print_log', default=5, type=int)
# parser.add_argument('--update_iteration', default=12, type=int)
parser.add_argument('--update_iteration', default=14, type=int)
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
script_name = os.path.basename(__file__)
# env = gym.make(args.env_name)
env = Maze

# actcri_net = 560
# actcri_net = 32
actcri_net = 128
actor_net = actcri_net
critic_net = actcri_net

if args.seed:
    # env.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

state_dim = 47
# state_dim = 14
action_dim = 10
max_action = 1
# min_Val = torch.tensor(1e-7).float().to(device) # min value

directory = './exp' + script_name + args.env_name +'./'

class Replay_buffer():
    '''
    Code based on:
    https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
    Expects tuples of (state, next_state, action, reward, done)
    '''
    def __init__(self, max_size=args.capacity):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def push(self, data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        # print('ind:',ind)
        x, y, u, r, d = [], [], [], [], []

        for i in ind:
            X, Y, U, R, D = self.storage[i]
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))

        return np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, actor_net)
        self.l2 = nn.Linear(actor_net, actor_net)
        self.l3 = nn.Linear(actor_net, action_dim)

        self.max_action = max_action

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.max_action * torch.tanh(self.l3(x))
        return x


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, critic_net)
        self.l2 = nn.Linear(critic_net, critic_net)
        self.l3 = nn.Linear(critic_net, 1)

    def forward(self, x, u):
        x = F.relu(self.l1(torch.cat([x, u], 1)))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x


class DDPG(object):
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-6)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)
        self.replay_buffer = Replay_buffer()
        # self.writer = SummaryWriter(directory)

        self.num_critic_update_iteration = 0
        self.num_actor_update_iteration = 0
        self.num_training = 0

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def update(self):

        for it in range(args.update_iteration):
            # Sample replay buffer
            x, y, u, r, d = self.replay_buffer.sample(args.batch_size)
            state = torch.FloatTensor(x).to(device)
            action = torch.FloatTensor(u).to(device)
            next_state = torch.FloatTensor(y).to(device)
            done = torch.FloatTensor(1-d).to(device)
            reward = torch.FloatTensor(r).to(device)

            # Compute the target Q value
            target_Q = self.critic_target(next_state, self.actor_target(next_state))
            target_Q = reward + (done * args.gamma * target_Q).detach()

            # Get current Q estimate
            current_Q = self.critic(state, action)

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q, target_Q)
            # self.writer.add_scalar('Loss/critic_loss', critic_loss, global_step=self.num_critic_update_iteration)
            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Compute actor loss
            actor_loss = -self.critic(state, self.actor(state)).mean()
            # self.writer.add_scalar('Loss/actor_loss', actor_loss, global_step=self.num_actor_update_iteration)

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            self.num_actor_update_iteration += 1
            self.num_critic_update_iteration += 1



def main():

    agent = DDPG(state_dim, action_dim, max_action)

    step = 0
    max_episode = 1000

    average_return1 = []
    average_return2 = []
    average_return3 = []
    average_return4 = []

    # step1 = 0
    p_j = 0

    # channel_mat = np.random.randint(0, 4, (3, 4))
    for i in range(max_episode):
        print('episode:', i)
        total_reward = 0
        step1 = 0

        cum_u1 = 0


        rit = np.random.poisson(26, 12)


        # step1 = 0
        cumulative_reward1 = 0
        cumulative_reward2 = 0
        cumulative_reward3 = 0
        cumulative_reward4 = 0

        uav_1_00 = 0
        uav_1_01 = 0
        uav_2_00 = 0
        uav_2_01 = 0



        observation = np.hstack((0, 0, 100, 1000, 1000, 100, -1000, -1000,
                         760, 380, 270, 160, 500, 500, 570, 120, 470, 600,
                         180, 800, 720, 280, 880, 820, 150, 200, 270, 580, 900, 460, 220, 380,
                         500, 350, 300, 650, 560, 750, 200, 680, 570, 450, 0,
                         15, 20, 25, 20))

        uav_center2 = np.array([[0, 0], [1000, 1000]])
        uav_center3 = np.array([[0, 0, 100], [1000, 1000, 100]])

        user_center2 = np.array([[180, 800], [720, 280], [880, 820], [150, 200], [270, 580],
                                [900, 460], [220, 380], [500, 350], [300, 650], [560, 750],
                                [200, 680], [570, 450]])
        user_center3 = np.array([[180, 800, 0], [720, 280, 0], [880, 820, 0], [150, 200, 0], [270, 580, 0],
                                [900, 460, 0], [220, 380, 0], [500, 350, 0], [300, 650, 0], [560, 750, 0],
                                [200, 680, 0], [570, 450, 0]])

        userte_center2 = np.array([[760, 380], [270, 160], [500, 500], [570, 120], [470, 600]])
        userte_center3 = np.array([[760, 380, 0], [270, 160, 0], [500, 500, 0], [570, 120, 0], [470, 600, 0]])

        jam_center2 = np.array([[-1000, -1000]])
        jam_center3 = np.array([[-1000, -1000, 0]])

        while True:

            action = agent.select_action(observation / 100)
            action = (action + np.random.normal(0, args.exploration_noise, size=10)).clip(-1, 1)


            observation_, reward, done, uav_center2, user_center2, rit, \
                jam_center2, uav_center3, user_center3, varpi_sum, t_sum, tsum_nsc\
                = env.step(action, uav_center2, step1, user_center2, rit, jam_center2,
                           uav_center3, user_center3, userte_center2, i)


            cumulative_reward1 += reward
            cumulative_reward2 += varpi_sum
            cumulative_reward3 += t_sum
            cumulative_reward4 += tsum_nsc


            agent.replay_buffer.push((observation / 100, observation_ / 100, action, reward, float(done)))

            observation = observation_


            if done:
                average_return1.append(cumulative_reward1)
                average_return2.append(cumulative_reward2)
                average_return3.append(cumulative_reward3)
                average_return4.append(cumulative_reward4)
                break

            step += 1
            step1 += 1


        print('cumulative_reward1:', cumulative_reward1)
        print('rit:', rit)

        # agent.update(12)
        agent.update()



    file_name = 'DQNr2.mat'
    savemat(file_name, {'DQNr2': average_return1})
    file_name = 'DQNr2_s.mat'
    savemat(file_name, {'DQNr2_s': average_return2})
    file_name = 'DQNr2_t.mat'
    savemat(file_name, {'DQNr2_t': average_return3})


    plt.figure(1)
    # plt.plot(np.arange(len(average_return)), average_return, 'r', marker='o')
    plt.plot(np.linspace(0, len(average_return1), max_episode), average_return1, label='BatteryLevel1')
    # plt.plot(np.linspace(0, len(average_return2), 1000), average_return2, label='BatteryLevel2')
    # plt.plot(np.linspace(0, len(average_return3), 1000), average_return3, label='BatteryLevel3')
    plt.legend(loc=4)
    plt.ylabel('Average_Return')
    plt.xlabel('training episodes')
    plt.show()

    end = time.time()
    print("game over!")
    print('运行时间:', end - start)
    engine = pyttsx3.init()
    engine.say('程序运行完成')
    engine.runAndWait()


if __name__ == '__main__':
    main()