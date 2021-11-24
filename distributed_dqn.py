from memory_remote import ReplayBuffer_remote
from dqn_model import _DQNModel
import torch
from custom_cartpole import CartPoleEnv
import gym
import torch
import time
import os
import ray
import numpy as np

from tqdm import tqdm
from random import uniform, randint

import io
import base64
from IPython.display import HTML

from dqn_model import DQNModel

import matplotlib.pyplot as plt

os.chdir(os.path.dirname(os.path.abspath(__file__)))

FloatTensor = torch.FloatTensor

ray.shutdown()
ray.init(include_webui=False, ignore_reinit_error=True, redis_max_memory=500000000, object_store_memory=5000000000)

# Set the Env name and action space for CartPole
ENV_NAME = 'CartPole-v0'
# Move left, Move right
ACTION_DICT = {
    "LEFT": 0,
    "RIGHT":1
}
# Register the environment
env_CartPole = gym.make(ENV_NAME)

hyperparams_CartPole = {
    'epsilon_decay_steps' : 100000, 
    'final_epsilon' : 0.1,
    'batch_size' : 32, 
    'update_steps' : 10, 
    'memory_size' : 2000, 
    'beta' : 0.99, 
    'model_replace_freq' : 2000,
    'learning_rate' : 0.0003,
    'use_target_model': True
}

@ray.remote
class Model_Server(object):
    def __init__(self, hyper_params, state, action_space=len(ACTION_DICT)) -> None:
        super().__init__()
        input_len = len(state)
        self.action_space = action_space
        output_len = action_space
        self.eval_network = DQNModel(input_len, output_len, learning_rate = hyper_params['learning_rate'])
        self.use_target_model = hyper_params['use_target_model']
        if self.use_target_model:
            self.target_model = DQNModel(input_len, output_len)
        self.batch_size = hyper_params['batch_size']
        self.update_steps = hyper_params['update_steps']
        self.model_replace_freq = hyper_params['model_replace_freq']
        self.beta = hyper_params['beta']
        self.initial_epsilon = 1
        self.final_epsilon = hyper_params['final_epsilon']
        self.epsilon_decay_steps = hyper_params['epsilon_decay_steps']
        self.episodes = 0

    def increase_episodes(self):
        self.episodes += 1

    def linear_decrease(self, initial_value, final_value, curr_steps, final_decay_steps):
        decay_rate = curr_steps / final_decay_steps
        if decay_rate > 1:
            decay_rate = 1
        return initial_value - (initial_value - final_value) * decay_rate
    
    def explore_or_exploit_policy(self, state, curr_steps):
        p = uniform(0, 1)
        # Get decreased epsilon
        epsilon = self.linear_decrease(self.initial_epsilon, 
                               self.final_epsilon,
                               self.steps,
                               self.epsilon_decay_steps)
        
        if p < epsilon:
            #return action
            return randint(0, self.action_space - 1)
        else:
            #return action
            return self.greedy_policy(state)
        
    def greedy_policy(self, state):
        return self.eval_model.predict(state)




@ray.remote
class Memory_Server(object):
    def __init__(self, hyper_params) -> None:
        super().__init__()
        self.memory = ReplayBuffer_remote(hyper_params['memory_size'])
        self.batch_size = hyper_params['batch_size']
        self.stored_expriences = 0

    def ask_for_batch(self):
        if self.stored_expriences > self.batch_size:
            return self.memory.sample.remote(self.batch_size)
        return


@ray.remote
class Collector(object):
    def __init__(self, model_server, env, memory_server, hyper_params) -> None:
        super().__init__()
        self.model_server = model_server
        self.env = env
        self.memory_server = memory_server
        self.steps = 0
        self.update_steps = hyper_params['update_steps']
        self.model_replace_freq = hyper_params['model_replace_freq']

    def generate_experiiences(self):

    

        
