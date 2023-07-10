import numpy as np
import torch
from torch import nn


class ActorModel(nn.Module):
    def __init__(self, obs_dim, act_dim, action_bound):
        super(ActorModel, self).__init__()
        hidden_dim = 256
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, act_dim)
        self.action_bound = torch.FloatTensor(action_bound) #对动作进行约束

    def forward(self, obs):
        # obs=torch.tensor(obs, dtype=torch.float32)
        hidden1 = torch.relu(self.fc1(obs))
        hidden2 = torch.relu(self.fc2(hidden1))
        action = torch.tanh(self.fc3(hidden2))
        scaled_action=action*self.action_bound # 对action进行放缩，实际上a in [-1,1]
        return scaled_action

# Critic网络输入的除了当前的state还有Actor输出的action，然后输出的是Q-value，即 Q(s,a)
class CriticModel(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(CriticModel, self).__init__()
        hidden_dim = 256
        self.fc1 = nn.Linear(obs_dim + act_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, obs, act):
        # 输入 state, action, 输出对应的Q(s,a)
        concat_vec = torch.concat([obs, act], axis=1)
        hidden1 = torch.relu(self.fc1(concat_vec))
        hidden2 = torch.relu(self.fc2(hidden1))
        Q = self.fc3(hidden2)
        return Q

class MYDDPG(nn.Module):
    def __init__(self, state_dim, action_dim, action_bound,tau=0.01,  memory_capacticy=1000, gamma=0.9, lr_a=0.001,
                 lr_c=0.002, batch_size=32):
        super(MYDDPG, self).__init__()
        self.state_dim = state_dim #观察维度
        self.action_dim = action_dim
        self.memory_capacticy = memory_capacticy
        self.t_replace_counter = 0
        self.tau=tau
        self.gamma = gamma
        self.lr_a = lr_a
        self.lr_c = lr_c
        self.batch_size = batch_size
        # 记忆库
        self.memory = np.zeros((memory_capacticy, state_dim * 2 + action_dim + 1))
        self.pointer = 0
        # 初始化 Actor 网络
        self.actor = ActorModel(state_dim, action_dim, action_bound)
        self.actor_target = ActorModel(state_dim, action_dim, action_bound)
        # 初始化 Critic 网络
        self.critic = CriticModel(state_dim, action_dim)
        self.critic_target = CriticModel(state_dim, action_dim)
        # 定义优化器
        self.aopt = torch.optim.AdamW(self.actor.parameters(), lr=lr_a)
        self.copt = torch.optim.AdamW(self.critic.parameters(), lr=lr_c)
        # 选取损失函数
        self.mse_loss = nn.MSELoss()

    # 存储序列数据
    def store_transition(self, s, a, r, s_next):
        transition = np.hstack((s, a, [r], s_next))
        index = self.pointer % self.memory_capacticy
        self.memory[index, :] = transition
        self.pointer += 1


    #先进性随机抽取,不是每次预测后都更新，是积累到一定次数 再更新
    def sample(self):
        indices = np.random.choice(self.memory_capacticy, size=self.batch_size)
        return self.memory[indices, :]

    #在选择动作
    def choose_action(self, s):
        # s = torch.unsqueeze(torch.FloatTensor(s),dim=0)
        s = torch.FloatTensor(s)
        action = self.actor(s)
        return action.detach().numpy()

    #采用软连接进行更新, 将selfmodel 数据更新到 targetmodel上
    def sync_parameter(self):
        a_layers=self.actor_target.named_children()
        c_layers=self.critic_target.named_children()

        # for循环进行更新
        for al in a_layers:
            al[1].weight.data.mul_((1 - self.tau))
            al[1].weight.data.add_(self.tau * self.actor.state_dict()[al[0] + '.weight'])
            al[1].bias.data.mul_((1 - self.tau))
            al[1].bias.data.add_(self.tau * self.actor.state_dict()[al[0] + '.bias'])
        for cl in c_layers:
            cl[1].weight.data.mul_((1 - self.tau))
            cl[1].weight.data.add_(self.tau * self.critic.state_dict()[cl[0] + '.weight'])
            cl[1].bias.data.mul_((1 - self.tau))
            cl[1].bias.data.add_(self.tau * self.critic.state_dict()[cl[0] + '.bias'])



    #actor 和critic 学习
    def learn(self):

        #将原来的 model 同步到target中
        self.sync_parameter()

        #从记忆库中sample数据
        batch_memory_sample=self.sample()
        batch_state = torch.FloatTensor(batch_memory_sample[:, :self.state_dim])
        batch_action=torch.FloatTensor(batch_memory_sample[:, self.state_dim:self.state_dim + self.action_dim])
        batch_reward = torch.FloatTensor(batch_memory_sample[:, -self.state_dim - 1: -self.state_dim])
        batch_state_next = torch.FloatTensor(batch_memory_sample[:,-self.state_dim:])

        #对两个base的a  c进行训练
        #对actor
        base_a=self.actor(batch_state)
        base_Q=self.critic(batch_state,base_a)
        a_loss=-torch.mean(base_Q)
        self.aopt.zero_grad()
        a_loss.backward(retain_graph=True)
        self.aopt.step()

        #对critic
        target_a_next = self.actor_target(batch_state_next)
        target_Q_next = self.critic_target(batch_state_next, target_a_next)
        target_Q = batch_reward + self.gamma * target_Q_next
        q_eval = self.critic(batch_state, batch_action)
        td_error = self.mse_loss(target_Q,q_eval) #td_error
        self.copt.zero_grad()
        td_error.backward()
        self.copt.step()




