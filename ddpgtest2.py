import time

import gym


#####################  hyper parameters  ####################
import numpy as np
import torch

from model.MYDDPG import MYDDPG

EPISODES = 200
EP_STEPS = 200
LR_ACTOR = 0.001
LR_CRITIC = 0.002
GAMMA = 0.9
TAU = 0.01
MEMORY_CAPACITY = 10000
BATCH_SIZE = 32
RENDER = False
ENV_NAME = 'Pendulum-v1'
var = 3.0 # the controller of exploration which will decay during training process


env = gym.make(ENV_NAME, render_mode="rgb_array")
env = env.unwrapped

s_dim = env.observation_space.shape[0]
a_dim = env.action_space.shape[0]
a_bound = env.action_space.high
a_low_bound = env.action_space.low

ddpgAgent=MYDDPG(state_dim=s_dim,action_dim=a_dim,action_bound=a_bound)

t1 = time.time()
for i in range(EPISODES):
    s,_ = env.reset()
    ep_r = 0
    for j in range(EP_STEPS):
        if RENDER:
            env.render()
        # add explorative noise to action
        a = ddpgAgent.choose_action(s)
        a = np.clip(np.random.normal(a, var), a_low_bound, a_bound) #添加噪音

        s_next, r, done, info,_ = env.step(a)
        ddpgAgent.store_transition(s, a, r / 10, s_next)  # store the transition to memory

        if ddpgAgent.pointer > MEMORY_CAPACITY:
            var *= 0.9995  # decay the exploration controller factor
            ddpgAgent.learn()

        s = s_next
        ep_r += r
        if j == EP_STEPS - 1:#训练次数够了，跳出这个,进行下个循环开始前 将展示
            print('Episode: ', i, ' Reward: %i' % (ep_r), 'Explore: %.2f' % var)
            if ep_r > -300:
                RENDER = True
            break

print('Running time: ', time.time() - t1)

