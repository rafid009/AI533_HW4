from torch.optim.adam import Adam
from memory_remote import ReplayBuffer_remote
from dqn_model import _DQNModel
import torch
from custom_cartpole import CartPoleEnv

import torch
import time
import os
import ray
import numpy as np

from tqdm import tqdm
from random import uniform, randint


import matplotlib.pyplot as plt
from filelock import FileLock

import torch.nn as nn

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Set the Env name and action space for CartPole
ENV_NAME = 'CartPole_distributed'

# Set result saveing floder
result_floder = ENV_NAME + "_distributed"
result_file = ENV_NAME + "/results.txt"
if not os.path.isdir(result_floder):
    os.mkdir(result_floder)
torch.set_num_threads(12)

FloatTensor = torch.FloatTensor

ray.shutdown()
ray.init(include_webui=False, ignore_reinit_error=True, redis_max_memory=500000000, object_store_memory=5000000000)

# Move left, Move right
ACTION_DICT = {
    "LEFT": 0,
    "RIGHT":1
}

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

start = time.time()

@ray.remote
class Model_Server(object):
    def __init__(self, hyper_params, input_len, memory, training_episodes, test_interval, action_space=len(ACTION_DICT)):
        super().__init__()
        self.action_space = action_space
        output_len = action_space

        self.eval_model = nn.DataParallel(_DQNModel(input_len, output_len))
        self.optimizer = Adam(self.eval_model.parameters(), lr = hyper_params['learning_rate'])
        self.loss_fn = nn.SmoothL1Loss()

        self.use_target_model = hyper_params['use_target_model']
        if self.use_target_model:
            self.target_model = nn.DataParallel(_DQNModel(input_len, output_len))

        self.prev_eval_models = []

        self.batch_size = hyper_params['batch_size']
        self.update_steps = hyper_params['update_steps']
        self.model_replace_freq = hyper_params['model_replace_freq']
        
        self.beta = hyper_params['beta']
        self.initial_epsilon = 1
        self.final_epsilon = hyper_params['final_epsilon']
        self.epsilon_decay_steps = hyper_params['epsilon_decay_steps']
        
        self.episodes = 0
        self.prev_episode = 0

        self.steps = 0
        self.prev_update = 0
        self.prev_replace = 0
        
        self.updates_made = 0
        
        # memory server
        self.memory = memory

        self.training_episodes = training_episodes
        self.test_interval = test_interval
        self.total_test_num = self.training_episodes // self.test_interval

        self.best_reward = 0
        self.results = [0] * (self.total_test_num + 1)
        self.result_counts = 0
        self.eval_ends = False

    # model stuffs
    def predict(self, input, model):
        input = FloatTensor(input).unsqueeze(0)
        q_values = model(input)
        action = torch.argmax(q_values)

        return action.item()
    
    def predict_batch(self, input, model):
        input = FloatTensor(input)
        q_values = model(input)
        values, q_actions = q_values.max(1)
        
        return q_actions, q_values

    def fit(self, q_values, target_q_values, optimizer, loss_fn):
        loss = loss_fn(q_values, target_q_values)  
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        self.updates_made += 1
        
    def replace(self):
        self.target_model.load_state_dict(self.eval_model.state_dict())
    
    def save(self, path):
        torch.save(self.eval_model.state_dict(), path + '/best_model.pt')
        
    def load(self, path):
        self.eval_model.load_state_dict(torch.load(path + '/best_model.pt'))


    ## agent stuffs
    def increase_episodes(self):
        self.episodes += 1

        if self.episodes // self.test_interval + 1 > len(self.prev_eval_models): 
            print('collector episode: ', self.episodes, ' time: ', time.time() - start)
            self.prev_eval_models.append(self.eval_model)
            
        if self.episodes >= self.training_episodes:
            return True
        return False

    def increase_steps(self):
        self.steps += 1
        return self.steps

    def get_best_reward(self):
        return self.best_reward

    def linear_decrease(self, initial_value, final_value, curr_steps, final_decay_steps):
        decay_rate = curr_steps / final_decay_steps
        if decay_rate > 1:
            decay_rate = 1
        return initial_value - (initial_value - final_value) * decay_rate
    
    def explore_or_exploit_policy(self, state):
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
        return self.predict(state, self.eval_model)

    def update(self):
        self.steps += 1
        if self.steps % self.update_steps == 0 and  self.steps > self.prev_update and \
        ray.get(self.memory.get_length.remote()) >= self.batch_size:
            self.prev_update = self.steps
            batch = ray.get(self.memory.sample.remote(self.batch_size))
            # print('update ', self.steps)
            (states, actions, reward, next_states,
            is_terminal) = batch
            
            states = states
            next_states = next_states
            terminal = FloatTensor([1 if t else 0 for t in is_terminal])
            reward = FloatTensor(reward)
            batch_index = torch.arange(self.batch_size,
                                    dtype=torch.long)
            
            # Current Q Values
            _, q_values = self.predict_batch(states, self.eval_model)
            q_values = q_values[batch_index, actions]
            
            # Calculate target
            if self.use_target_model:
                actions, q_next = self.predict_batch(next_states, self.target_model)
            else:
                actions, q_next = self.predict_batch(next_states, self.eval_model)
                
            #INSERT YOUR CODE HERE --- neet to compute 'q_targets' used below
            q_target = (terminal * reward) + (1.0 - terminal) * (reward + (self.beta * torch.max(q_next, 1)[0]))

            # update model
            self.fit(q_values, q_target, self.optimizer, self.loss_fn)

        if self.use_target_model and self.steps % self.model_replace_freq == 0 and self.steps > self.prev_replace:
            self.prev_replace = self.steps
            self.replace()

    def add_result(self, result, num):
        self.results[num] = result

    def get_results(self):
        return self.results

    def ask_evaluation(self):
        if len(self.prev_eval_models) > self.result_counts:
            num = self.result_counts
            eval_model = self.prev_eval_models[num]
            self.result_counts += 1
        #if self.episodes != 0 and self.episodes % self.test_intervals == 0 and self.episodes > self.prev_episode:
            print('episode ', self.episodes, ' time: ', time.time()-start)
            print('ask')
            return eval_model, False, num
        else:
            if self.episodes >= self.training_episodes:
                self.eval_ends = True
            return None, self.eval_ends, None

    def write_result_to_file(self, result_file, avg_result):
        # Create a filelock object. Consider using an absolute path for the lock.
        if avg_result >= self.best_reward:
            self.best_reward= avg_result
            self.save(result_floder)
        with FileLock(result_file):
            with open(result_file, "a+") as f:
                f.write(str(avg_result) + "\n")
                f.close()

@ray.remote
class Collector(object):
    def __init__(self, model_server, env, memory, update_steps):
        super().__init__()
        self.model_server = model_server
        self.env = env
        self.max_episode_steps = env._max_episode_steps
        self.memory = memory
        self.update_steps = update_steps

    def generate_experiences(self):
        while True:
            state = self.env.reset()
            done = False
            steps = 0
            # print('collector')
            while steps < self.max_episode_steps and not done:
                #INSERT YOUR CODE HERE
                # add experience from explore-exploit policy to memory
                # update the model every 'update_steps' of experience
                # update the target network (if the target network is being used) every 'model_replace_freq' of experiences 
                action = ray.get(self.model_server.explore_or_exploit_policy.remote(state))
                steps += 1

                next_state, reward, done, info = self.env.step(action)
                
                self.memory.add.remote(state, action, reward, next_state, done)
                state = next_state
                self.model_server.update.remote()

                # ray.get(self.model_server.update.remote())
            do_continue = ray.get(self.model_server.increase_episodes.remote())
            if do_continue:
                break
    

@ray.remote
class Evaluator(object):
    def __init__(self, model_server, env, trials=30):
        super().__init__()
        self.env = env
        self.trials = trials
        self.max_episode_steps = env._max_episode_steps
        self.model_server = model_server

    def greedy_policy(self, state, model):
        return self.model_server.predict.remote(state, model)

    def evaluate(self):
        while True:
            eval_model, eval_ends, num = ray.get(self.model_server.ask_evaluation.remote())
            if eval_ends:
                break
            if eval_model is None:
                continue
            print('eval')
            total_reward = 0
            for i in range(self.trials):
                state = self.env.reset()
                done = False
                steps = 0

                while steps < self.max_episode_steps and not done:
                    steps += 1
                    action = ray.get(self.model_server.greedy_policy.remote(state))
                    state, reward, done, _ = self.env.step(action)
                    total_reward += reward

            avg_reward = total_reward / self.trials
            self.model_server.add_result.remote(avg_reward, num)
            print('reward: ', avg_reward)
            # self.model_server.write_result_to_file.remote(result_file, avg_reward)
            
    
    # def save_model(self):
    #     self.eval_model.save(result_floder + '/best_model.pt')

    # def load_model(self):
    #     self.eval_model.load(result_floder + '/best_model.pt')
    
def plot_result(total_rewards ,learning_num, legend):
    print("\nLearning Performance:\n")
    episodes = []
    for i in range(len(total_rewards)):
        episodes.append(i * learning_num + 1)
        
    plt.figure(num = 1)
    fig, ax = plt.subplots()
    plt.plot(episodes, total_rewards)
    plt.title('performance')
    plt.legend(legend)
    plt.xlabel("Episodes")
    plt.ylabel("total rewards")
    plt.savefig('dist-dqn.jpg')
    plt.close()

class distributed_DQN_agent(object):
    def __init__(self, cw_num, ew_num, input_len, training_episodes, test_intervals, eval_trials, hyper_params, action_space=len(ACTION_DICT)):
        super().__init__()
        self.memory = ReplayBuffer_remote.remote(hyper_params['memory_size'])
        self.model_server = Model_Server.remote(hyper_params, input_len, self.memory, training_episodes, test_intervals, action_space)
        self.cw_num = cw_num
        self.ew_num = ew_num
        self.eval_trials = eval_trials
        self.update_steps = hyper_params['update_steps']

    def learn_and_evaluate(self):
        workers_id = []
        for cw in range(self.cw_num):
            simulator = CartPoleEnv()
            collector = Collector.remote(self.model_server, simulator, self.memory, self.update_steps)
            workers_id.append(collector.generate_experiences.remote())
        for ew in range(self.ew_num):
            simulator = CartPoleEnv()
            evaluator = Evaluator.remote(self.model_server, simulator, self.eval_trials)
            workers_id.append(evaluator.evaluate.remote())
        ray.wait(workers_id, len(workers_id))
        return ray.get(self.model_server.get_results.remote())
        

cw_num = 4
ew_num = 4
training_episodes = 10000
test_intervals = 50
eval_trials = 30
action_space = len(ACTION_DICT)
simulator = CartPoleEnv()
input_len = len(simulator.reset())
dist_DQN = distributed_DQN_agent(cw_num, ew_num, input_len, training_episodes, test_intervals, eval_trials, hyperparams_CartPole, action_space)
result = dist_DQN.learn_and_evaluate()
end = time.time()
print('time: ', end-start, ' s')
plot_result(result, test_intervals, ["batch_update with target_model"])
