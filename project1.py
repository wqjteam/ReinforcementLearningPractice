import time

import gym
from gym import envs
env_list = envs.registry.keys()
env_ids = [env_item for env_item in env_list]
# print('There are {0} envs in gym'.format(len(env_ids)))
# print(env_ids)

'''
env = gym.make(id)
	说明：生成环境
	参数：Id(str类型)  环境ID
	返回值：env(Env类型)  环境

	环境ID是OpenAI Gym提供的环境的ID，可以通过上一节所述方式进行查看有哪些可用的环境
	例如，如果是“CartPole”环境，则ID可以用“CartPole-v1”。返回“Env”对象作为返回值
'''

env = gym.make('CartPole-v1')
print('观测空间 = {}'.format(env.observation_space))
print('动作空间 = {}'.format(env.action_space))
print('动作数 = {}'.format(env.action_space.n))
print('初始状态 = {}'.format(env.state))


'''
state = env.reset()
	说明：重置环境，回到初始状态
	返回值：state（object类型） 环境的最初状态。类型由属性“observation_space”决定
'''

init_state = env.reset()
print('初始状态 = {}'.format(init_state))
print('初始状态 = {}'.format(env.state))

'''
Observation, reward, terminated, truncated, info = env.step(action)
	说明：环境执行一步动作
	参数：action（object 类型） 动作
	返回值：results（tuple 类型） (下一状态，报酬，episode 是否完成，日志信息)

	将“动作”传递给环境，返回值返回“下一个状态”（object）、“报酬”（float）、“ episode 是否完成”（bool）、“日志信息”（dict）
	传递给环境的“动作”类型，由属性“action_space”决定
'''


for k in range(5):
    action = env.action_space.sample()
    state, reward, done, truncated, info = env.step(action)
    print('动作 = {0}: 当前状态 = {1}, 奖励 = {2}, 结束标志 = {3}, 日志信息 = {4}'.format(action, state, reward, done, info))



'''
env.render(mode='human')
	说明：渲染环境画面
	参数：mode（str类型） 渲染模式
	返回值：对应渲染模式的返回值
'''

'''
关闭环境
env.close()
'''
# env.close()

'''
    env.sample_space.sample(): 对动作空间进行随机采样
   动作的选择应该基于策略进行。但是，完全随机的选择动作也是一种策略，或者可以说是一种基线(baseline)策略。
   任何一种能够体现有效的学习效果的策略都不应该比这种基线策略的效果差。
   这就好比任何一个有效的预测（股票涨跌、球赛胜负啊随便什么的）算法不能比随机扔硬币决定要更差。
   如果一种基于一种习得的策略来选取动作其最终得到的回报不比以上随机采样策略好的话，就说明这个习得的策略没有任何价值。
'''


'''
env.seed(seed=None)
	说明：指定随机数种子
	参数：seed（int 类型） 随机种子
	返回值：seeds（list 类型） 在环境中使用的随机数种子列表
	用env.seed()指定环境的随机数种子。如果想要训练的再现性，或者想要根据不同的环境使用不同的随机数种子，就可以使用该方法
'''



'''
一种随机策略的演示
'''

# 生成环境
env = gym.make('CartPole-v1',render_mode='human')
# 环境初始化
state = env.reset()
# 循环交互
while True:
    # 渲染画面
    env.render()
    # 从动作空间随机获取一个动作
    action = env.action_space.sample()
    # agent与环境进行一步交互
    state, reward, terminated,truncated, info = env.step(action)
    print('state = {0}; reward = {1}'.format(state, reward))
    # 判断当前episode 是否完成
    if done:
        print('done')
        break
    time.sleep(1)
# 环境结束
env.close()