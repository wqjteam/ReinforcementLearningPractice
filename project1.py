import gym
from gym import envs
env_list = envs.registry.keys()
env_ids = [env_item for env_item in env_list]
print('There are {0} envs in gym'.format(len(env_ids)))
print(env_ids)