import os
import time

import numpy as np

import parl
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from parl.utils import logger
# 将神经网络输出映射到对应的 实际动作取值范围 内

from parl.utils import ReplayMemory  # 经验回放

from rlschool import make_env  # 使用 RLSchool 创建飞行器环境

from model.DDPG import DDPG
from model.MYDDPG import MYDDPG
from util import action_mapping

ACTOR_LR = 0.0002  # Actor网络更新的 learning rate
CRITIC_LR = 0.001  # Critic网络更新的 learning rate

GAMMA = 0.99  # reward 的衰减因子，一般取 0.9 到 0.999 不等
TAU = 0.001  # target_model 跟 model 同步参数 的 软更新参数
MEMORY_SIZE = 1000  # replay memory的大小，越大越占用内存
REWARD_SCALE = 0.01  # reward 的缩放因子
BATCH_SIZE = 256  # 每次给agent learn的数据数量，从replay memory随机里sample一批数据出来
EPISODES = 100000  # 总训练步数
EP_STEPS = 1000  # 每个N步评估一下算法效果，每次评估5个episode求平均reward
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 评估 agent, 跑 5 个episode，总reward求平均
def evaluate(env, agent):
    eval_reward = []
    for i in range(5):
        obs = env.reset()
        total_reward = 0
        while True:
            env.render()
            a = agent.choose_action(obs)
            s_next, r, done, info = env.step(a)
            obs = s_next
            total_reward += r

            if done:
                break
        eval_reward.append(total_reward)
    return np.mean(eval_reward)


# 创建飞行器环境
env = make_env("Quadrotor", task="hovering_control")
env.reset()
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]
low_bound = env.action_space.low[0]
high_bound = env.action_space.high[0]
var = high_bound

agent = MYDDPG(state_dim=obs_dim, action_dim=act_dim, action_bound=[high_bound], batch_size=BATCH_SIZE,
               memory_capacticy=MEMORY_SIZE)

if __name__ == '__main__':
    # 启动训练

    for i in range(EPISODES):
        ep_r = 0
        s = env.reset()
        for j in range(EP_STEPS):
            a = agent.choose_action(s)
            a = np.clip(np.random.normal(a, high_bound), low_bound, high_bound)
            s_next, r, done, info = env.step(a)
            # logger.info('EPISODES {},STEP {}, Test reward: {}'.format(i, j, r))
            agent.store_transition(s, a, r / 10, s_next)

            if agent.pointer > MEMORY_SIZE:
                var *= 0.9995  # decay the exploration controller factor
                agent.learn()
            s = s_next
            ep_r += r
        # 测试一轮
        if i // 5 == 0:
            evaluate_reward = evaluate(env, agent)
            logger.info('EPISODES {}, Test reward: {}'.format(i, evaluate_reward))  # 打印评估的reward

            # 每评估一次，就保存一次模型，以训练的step数命名
            ckpt = 'model_dir/steps_{}.ckpt'.format(i)
            torch.save(agent.state_dict(), ckpt)

######################################################################
######################################################################
#
# 7. 请选择你训练的最好的一次模型文件做评估
#
######################################################################
######################################################################
ckpt = 'model_dir/steps_??????.ckpt'  # 请设置ckpt为你训练中效果最好的一次评估保存的模型文件名称
agent.load_state_dict(torch.load(ckpt))

env.render()
evaluate_reward = evaluate(env, agent)
logger.info('Evaluate reward: {}'.format(evaluate_reward))  # 打印评估的reward
