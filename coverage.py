from ray import tune
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from pettingzoo.mpe import simple_spread_v2
import torch
from torch import nn
from pettingzoo.utils import random_demo
import numpy as np

def env_creator(**kwargs):
    env = simple_spread_v2.env(N=kwargs['n'], local_ratio=0.5, max_cycles=50, continuous_actions=True)
    return env


if __name__ == '__main__':
    env = env_creator(n=3)
    env.reset()
    real_env = env.env.env
    i = 1
    for agent in env.agent_iter():
        observation, reward, done, info = env.last()

        print(observation, reward, done, info)
        env.step([(n + 1) * (10**(-i)) for n in range(5)])
        i += 1