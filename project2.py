import torch
import torch.nn as nn
from collections import deque
import numpy as np
import gym
import random
from net import AtariNet
from util import preprocess
import matplotlib.pyplot as plt #  plt 用于显示图片

BATCH_SIZE = 32
LR = 0.001
START_EPSILON = 1.0             #  贪婪程度随时间变化，前期尽可能贪婪，后期弱化
FINAL_EPSILON = 0.1
EPSILON = START_EPSILON
EXPLORE = 1000000
GAMMA = 0.99                    # 折扣比例
TOTAL_EPISODES = 10000000
MEMORY_SIZE = 1000000
MEMORY_THRESHOLD = 100000
UPDATE_TIME = 10000
TEST_FREQUENCY = 1
env = gym.make('Pong-v0')
env = env.unwrapped             # 打开包装
ACTIONS_SIZE = env.action_space.n      # action的个数
class Agent(object):
    def __init__(self):
        self.network, self.target_network = AtariNet(ACTIONS_SIZE), AtariNet(ACTIONS_SIZE)
        #  设计了两个策略，主网络（Action-Value）和目标网络（Target Action-Value)
        self.memory = deque()       # 啥都没有
        self.learning_count = 0
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=LR)
        #  优化器，优化的参数为主网络的相关参数,在这里定义了优化的网络
        self.loss_func = nn.MSELoss()
        # 使用的是MSE损失函数
    def action(self, state, israndom):
        if israndom and random.random() < EPSILON:              # 前期尽可能贪婪
            return np.random.randint(0, ACTIONS_SIZE)
            #  返回[0，ACTIONS_SIZE=6)中的任意数作为采取的动作，同时不执行下面的代码，
        state = torch.unsqueeze(torch.FloatTensor(state), 0)
        #  FloatTensor将numpy.array格式转化为tensor格式
        #  torch.unsqueeze()主要是用来扩展数据维度
        actions_value = self.network.forward(state)
        return torch.max(actions_value, 1)[1].data.numpy()[0]  # 输出最大值，可以看到通过输入状态，网络输出的是该状态下所有动作

    def learn(self, state, action, reward, next_state, done):
        if done:
            self.memory.append((state, action, reward, next_state, 0))
        else:
            self.memory.append((state, action, reward, next_state, 1))    # 括号内的算成整体来计算个数
        if len(self.memory) > MEMORY_SIZE:
            self.memory.popleft()      # popleft是用来移除最左端一个数据
        if len(self.memory) < MEMORY_THRESHOLD:
            return
        #  这一部分应该是用来存放memory_size个数据的经验数据池,同时对数据的大小有一定的限制

        if self.learning_count % UPDATE_TIME == 0:
            self.target_network.load_state_dict(self.network.state_dict())
            #  state_dict是用来存放训练过程中的神经网络的权重参数
            #  过一段时间进行更新同步主网络
        self.learning_count += 1
        #  计数，这一部分是隔一段时间将目标网络从主网络复制到主网络

        # random choose batch_size sample and 对应进行分类
        state = torch.FloatTensor([x[0] for x in batch])
        action = torch.LongTensor([[x[1]] for x in batch])
        reward = torch.FloatTensor([[x[2]] for x in batch])
        next_state = torch.FloatTensor([x[3] for x in batch])
        done = torch.FloatTensor([[x[4]] for x in batch])


        eval_q = self.network.forward(state).gather(1, action) # 由主网络产生当前状态价值函数
        next_q = self.target_network(next_state).detach()       #由目标网络生成
        target_q = reward + GAMMA * next_q.max(1)[0].view(BATCH_SIZE, 1) * done
        loss = self.loss_func(eval_q, target_q)

        # 常规套路进行梯度下降改进相关参数，可以看到没有在这里确定优化的网络是谁，而是在最开始的定义里有
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

agent = Agent()
j=0;
for i_episode in range(TOTAL_EPISODES):
    #图片的输入
    state = env.reset()                 # 将环境恢复,初始大小为210*160*3三通道彩色图片
    #  plt.figure(1)
    #  plt.imshow(state)                   # 自己写的输出此时env的图像
    state = preprocess(state)           # preprocess函数是另外文件中的函数，见util.py,对图片尺寸进行处理实现
    while True:                         # 无线循环，直到break停止
        # env.render()
        j+=1
        action = agent.action(state, True)
        next_state, reward, done, info = env.step(action)     # 这里输出的done代表游戏是否结束
        #  plt.figure(2)
        #  plt.imshow(next_state)
        next_state = preprocess(next_state)
        agent.learn(state, action, reward, next_state, done)

        state = next_state
        if done:
            break
    print(j)
    if EPSILON > FINAL_EPSILON:
        EPSILON -= (START_EPSILON - FINAL_EPSILON) / EXPLORE     # 贪婪概率不断减小
  #  TEST 隔一段时间对网络进行测试，并输出用网络输出的reward结果
    if i_episode % TEST_FREQUENCY == 0:
        state = env.reset()
        state = preprocess(state)
        total_reward = 0
        while True:
            env.render()
            action = agent.action(state, israndom=False)
            next_state, reward, done, info = env.step(action)
            next_state = preprocess(next_state)

            total_reward += reward

            state = next_state
            if done:
                break
        print('episode: {} , total_reward: {}'.format(i_episode, round(total_reward, 3)))
 env.close()
