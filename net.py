import torch
from torch import nn


class AtariNet(nn.Module):

    def __init__(self, num_actions):
        super(AtariNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),                  #第一层  卷积核32个，大小
            nn.ReLU()                                                   #RELU激励函数
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.hidden = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512, bias=True),
            nn.ReLU()
        )
        self.out = nn.Sequential(
            nn.Linear(512, num_actions, bias=True)
        )
        #84*84*1——>conv1——>20*20*32——>conv2——>9*9*64——>conv3——>7*7*64——>
        # FC1——>512——>FC2——num_action为采取的动作
        #以上是网络层的输入输出数据维度

        self.apply(self.init_weights) # 网络的参数初始化

    def init_weights(self, m):
        if type(m) == nn.Conv2d:
            m.weight.data.normal_(0.0, 0.02)
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
        # 分类初始化参数
        # type()函数返回类型

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)   #将输入的多维向量转为一维
        x = self.hidden(x)
        x = self.out(x)
        #输出的是1*6的向量，再取每列的最大值
        return x


class CnnDQN(nn.Module):
    def __init__(self, inputs_shape, num_actions):
        super(CnnDQN, self).__init__()

        self.inut_shape = inputs_shape
        self.num_actions = num_actions

        self.features = nn.Sequential(
            nn.Conv2d(inputs_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(self.features_size(), 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def features_size(self):
        return self.features(torch.zeros(1, *self.inut_shape)).view(1, -1).size(1)

