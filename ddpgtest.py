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
from util import action_mapping

ACTOR_LR = 0.0002  # Actor网络更新的 learning rate
CRITIC_LR = 0.001  # Critic网络更新的 learning rate

GAMMA = 0.99  # reward 的衰减因子，一般取 0.9 到 0.999 不等
TAU = 0.001  # target_model 跟 model 同步参数 的 软更新参数
MEMORY_SIZE = 1e6  # replay memory的大小，越大越占用内存
MEMORY_WARMUP_SIZE = 1e4  # replay_memory 里需要预存一些经验数据，再从里面sample一个batch的经验让agent去learn
REWARD_SCALE = 0.01  # reward 的缩放因子
BATCH_SIZE = 256  # 每次给agent learn的数据数量，从replay memory随机里sample一批数据出来
TRAIN_TOTAL_STEPS = 1e6  # 总训练步数
TEST_EVERY_STEPS = 1e4  # 每个N步评估一下算法效果，每次评估5个episode求平均reward
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ActorModel(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(ActorModel, self).__init__()
        ######################################################################
        ######################################################################
        #
        # 2. 请配置model结构
        #
        ######################################################################
        ######################################################################
        hidden_dim = 256
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, act_dim)

    def forward(self, obs):
        ######################################################################
        ######################################################################
        #
        # 3. 请组装policy网络
        #
        ######################################################################
        ######################################################################
        # obs=torch.tensor(obs, dtype=torch.float32)
        hidden1 = F.relu(self.fc1(obs))
        hidden2 = F.relu(self.fc2(hidden1))
        logits = F.tanh(self.fc3(hidden2))
        return logits


class CriticModel(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(CriticModel, self).__init__()
        ######################################################################
        ######################################################################
        #
        # 4. 请配置model结构
        #
        ######################################################################
        ######################################################################
        hidden_dim = 256
        self.fc1 = nn.Linear(obs_dim + act_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, obs, act):
        # 输入 state, action, 输出对应的Q(s,a)

        ######################################################################
        ######################################################################
        #
        # 5. 请组装Q网络
        #
        ######################################################################
        ######################################################################
        concat_vec = torch.concat([obs, act], axis=1)
        hidden1 = F.relu(self.fc1(concat_vec))
        hidden2 = F.relu(self.fc2(hidden1))
        Q = self.fc3(hidden2)
        return Q


class QuadrotorModel(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(QuadrotorModel, self).__init__()
        self.actor_model = ActorModel(obs_dim, act_dim)
        self.critic_model = CriticModel(obs_dim, act_dim)

    def policy(self, obs):
        return self.actor_model(obs)

    def value(self, obs, act):
        return self.critic_model(obs, act)

    def get_actor_params(self):
        return self.actor_model.parameters()

    def get_critic_params(self):
        return self.critic_model.parameters()


class QuadrotorAgent(nn.Module):
    def __init__(self, algorithm, obs_dim, act_dim=4):
        super(QuadrotorAgent, self).__init__()
        assert isinstance(obs_dim, int)
        assert isinstance(act_dim, int)
        self.alg = algorithm
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        # 注意，在最开始的时候，先完全同步target_model和model的参数
        self.alg.sync_target(decay=0)

    def predict(self, obs):
        obs = obs.unsqueeze(dim=0)
        pred_act = self.alg.predict(obs)
        return pred_act

    def learn(self, obs, act, reward, next_obs, terminal):
        _, critic_cost = self.alg.learn(obs, act, reward, next_obs, terminal)
        self.alg.sync_target()
        return critic_cost


def run_episode(env, agent, rpm):
    obs = env.reset()

    total_reward, steps = 0, 0
    while True:
        steps += 1
        batch_obs = torch.tensor(obs).unsqueeze(dim=0).to(device)
        action = agent.predict(batch_obs).to(device)
        action = torch.squeeze(action)

        # 给输出动作增加探索扰动，输出限制在 [-1.0, 1.0] 范围内
        action = torch.clip(torch.normal(action, 1.0), -1.0, 1.0)
        # 动作映射到对应的 实际动作取值范围 内, action_mapping是从parl.utils那里import进来的函数
        action = action_mapping(action, env.action_space.low[0], env.action_space.high[0])

        next_obs, reward, done, info = env.step(action.detach().cpu().numpy())
        rpm.append(obs, action.detach().cpu().numpy(), REWARD_SCALE * reward, next_obs, done)

        if rpm.size() > MEMORY_WARMUP_SIZE:
            batch_obs, batch_action, batch_reward, batch_next_obs, batch_terminal = rpm.sample_batch(BATCH_SIZE)
            critic_cost = agent.learn(torch.tensor(batch_obs).to(device),
                                      torch.tensor(batch_action).to(device),
                                      batch_reward,
                                      torch.tensor(batch_next_obs).to(device),
                                      batch_terminal)

        obs = next_obs
        total_reward += reward

        if done:
            break
    return total_reward, steps


# 评估 agent, 跑 5 个episode，总reward求平均
def evaluate(env, agent):
    eval_reward = []
    for i in range(5):
        obs = env.reset()
        total_reward, steps = 0, 0
        while True:
            env.render()
            # time.sleep(100)
            batch_obs = torch.tensor(obs).unsqueeze(dim=0).to(device)
            action = agent.predict(batch_obs).to(device)
            action = torch.squeeze(action)
            action = action_mapping(action, env.action_space.low[0], env.action_space.high[0])

            next_obs, reward, done, info = env.step(action.detach().cpu().numpy())

            obs = next_obs
            total_reward += reward
            steps += 1

            if done:
                break
        eval_reward.append(total_reward)
    return np.mean(eval_reward)


# 创建飞行器环境
env = make_env("Quadrotor", task="hovering_control")
env.reset()
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]

# 根据parl框架构建agent
######################################################################
######################################################################
#
# 6. 请构建agent:  QuadrotorModel, DDPG, QuadrotorAgent三者嵌套
#
######################################################################
######################################################################
model = QuadrotorModel(act_dim=act_dim, obs_dim=obs_dim)
algorithm = DDPG(model=model, gamma=GAMMA, tau=TAU, actor_lr=ACTOR_LR, critic_lr=CRITIC_LR)
agent = QuadrotorAgent(algorithm, obs_dim, act_dim=act_dim).to(device)

# parl库也为DDPG算法内置了ReplayMemory，可直接从 parl.utils 引入使用
rpm = ReplayMemory(int(MEMORY_SIZE), obs_dim, act_dim)

if __name__ == '__main__':
    # 启动训练
    test_flag = 0
    total_steps = 0
    while total_steps < TRAIN_TOTAL_STEPS:
        train_reward, steps = run_episode(env, agent, rpm)
        total_steps += steps
        # logger.info('Steps: {} Reward: {}'.format(total_steps, train_reward)) # 打印训练reward

        if total_steps // TEST_EVERY_STEPS >= test_flag:  # 每隔一定step数，评估一次模型
            while total_steps // TEST_EVERY_STEPS >= test_flag:
                test_flag += 1

            evaluate_reward = evaluate(env, agent)
            logger.info('Steps {}, Test reward: {}'.format(
                total_steps, evaluate_reward))  # 打印评估的reward

            # 每评估一次，就保存一次模型，以训练的step数命名
            ckpt = 'model_dir/steps_{}.ckpt'.format(total_steps)
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
