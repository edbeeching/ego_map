#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 09:54:26 2018

@author: anonymous

A class that can be used to implement many parallel environments

"""
import gc
import time
from itertools import islice



from collections import deque


import multiprocessing as mp
import numpy as np
try:
    from gym.spaces.box import Box
    from baselines.common.atari_wrappers import make_atari, wrap_deepmind
except ImportError:
    print('Unable to import gym / OpenAI baselines, I assume you are running the doom env')

if __name__ == '__main__':
    from arguments import parse_game_args  
    import timeit      
else:    
    from .arguments import parse_game_args        
from environments import DoomEnvironment

def worker(in_queue, out_queue, params, is_train, idx=0):
    env = DoomEnvironment(params, idx=idx, is_train=is_train, use_shaping=params.use_shaping)
    while True:
        action = in_queue.get()
        if action is None:
            break
        elif action == 'reset':
            out_queue.put(env.reset())
            
        elif action == 'depth_trim':
            out_queue.put(env.get_depth()[2:-2,2:-2])
            
        elif action == 'depth':
            out_queue.put(env.get_depth())
            
        elif action == 'ego_depth':
            out_queue.put(env.get_ego_depth())
            
        elif action == 'ego_depth_trim':
            out_queue.put(env.get_ego_depth()[2:-2,2:-2])      

        elif action == 'deltas':
            out_queue.put(env.get_player_deltas())
            
        elif action == 'positions':
            out_queue.put(env.get_player_position())
            
        elif action == 'origins':
            out_queue.put(env.get_player_origins())
            
        elif action == 'pos_deltas_origins':
            out_queue.put(env.get_player_pos_delta_origin())
            
        else:
            obs, reward, done, info = env.step(action)
            out_queue.put((obs, reward, done, info))


def pipe_worker(pipe, params, is_train, idx=0):
    env = DoomEnvironment(params, idx=idx, is_train=is_train, use_shaping=params.use_shaping)
    while True:
        action = pipe.recv()
        if action is None:
            break
        elif action == 'reset':
            pipe.send(env.reset())
            
        elif action == 'depth_trim':
            pipe.send(env.get_depth()[2:-2,2:-2])
            
        elif action == 'depth':
            pipe.send(env.get_depth())
            
        elif action == 'ego_depth':
            pipe.send(env.get_ego_depth())
            
        elif action == 'ego_depth_trim':
            pipe.send(env.get_ego_depth()[2:-2,2:-2])      

        elif action == 'deltas':
            pipe.send(env.get_player_deltas())
            
        elif action == 'positions':
            pipe.send(env.get_player_position())
            
        elif action == 'origins':
            pipe.send(env.get_player_origins())
            
        elif action == 'pos_deltas_origins':
            pipe.send(env.get_player_pos_delta_origin())
        
        elif action == 'loops':
            pipe.send(env.get_loop())            
        else:
            obs, reward, done, info = env.step(action)
            pipe.send((obs, reward, done, info))
            
def pipe_worker2(pipe, params, is_train, idx_range=[0]):

    envs_queue = deque()    
    for idx in idx_range:
        env = DoomEnvironment(params, idx=idx, is_train=is_train, use_shaping=params.use_shaping, fixed_scenario=True)        
        obs = env.reset()
        envs_queue.append((obs, env))
        
    obs, cur_env = envs_queue.pop()
    
    
    while True:
        action = pipe.recv()
        if action is None:
            break
        elif action == 'reset':
            pipe.send(cur_env.reset())
            
        elif action == 'depth_trim':
            pipe.send(cur_env.get_depth()[2:-2,2:-2])
            
        elif action == 'depth':
            pipe.send(cur_env.get_depth())
            
        elif action == 'ego_depth':
            pipe.send(cur_env.get_ego_depth())
            
        elif action == 'ego_depth_trim':
            pipe.send(cur_env.get_ego_depth()[2:-2,2:-2])      

        elif action == 'deltas':
            pipe.send(cur_env.get_player_deltas())
            
        elif action == 'positions':
            pipe.send(cur_env.get_player_position())
            
        elif action == 'origins':
            pipe.send(cur_env.get_player_origins())
            
        elif action == 'pos_deltas_origins':
            pipe.send(cur_env.get_player_pos_delta_origin())
        
        elif action == 'loops':
            pipe.send(cur_env.get_loop())
        
        else:
            obs, reward, done, info = cur_env.step(action)
            
            if done:
                envs_queue.append((obs, cur_env))
                obs, cur_env = envs_queue.popleft()
                
            pipe.send((obs, reward, done, info))
            

def worker2(in_queue, out_queue, params, is_train, idx_range=[0]):
    
    envs_queue = deque()    
    for idx in idx_range:
        env = DoomEnvironment(params, idx=idx, is_train=is_train, use_shaping=params.use_shaping, fixed_scenario=True)        
        obs = env.reset()
        envs_queue.append((obs, env))
        
    obs, cur_env = envs_queue.pop()
    
    while True:
        action = in_queue.get()
        if action is None:
            break
        elif action == 'reset':
            out_queue.put(cur_env.reset())
            
        elif action == 'depth_trim':
            out_queue.put(cur_env.get_depth()[2:-2,2:-2])
            
        elif action == 'depth':
            out_queue.put(cur_env.get_depth())
            
        elif action == 'ego_depth':
            out_queue.put(cur_env.get_ego_depth())
            
        elif action == 'ego_depth_trim':
            out_queue.put(cur_env.get_ego_depth()[2:-2,2:-2])      

        elif action == 'deltas':
            out_queue.put(cur_env.get_player_deltas())
            
        elif action == 'positions':
            out_queue.put(cur_env.get_player_position())
            
        elif action == 'origins':
            out_queue.put(cur_env.get_player_origins())
            
        elif action == 'pos_deltas_origins':
            out_queue.put(cur_env.get_player_pos_delta_origin())
        elif action == 'loops':
            out_queue.send(cur_env.get_loop())            
        else:
            obs, reward, done, info = cur_env.step(action)
            
            if done:
                envs_queue.append((obs, cur_env))
                obs, cur_env = envs_queue.popleft()
                
            out_queue.put((obs, reward, done, info))

            
def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())    

class MultiEnvsMP(object):
    """
        Run many envs on different processes to speed up simulation.
        
        Here this is fixed to be 16 workers but this could be increased if more
        compute is available
    """
    def __init__(self, env_id, num_envs, num_processes, params, is_train=True):
        
        
        self.in_queues = [mp.Queue(maxsize=8) for _ in range(num_envs)]
        self.out_queues = [mp.Queue(maxsize=8) for _ in range(num_envs)]
        self.workers = []
        
        if params.fixed_scenario:
            if is_train:
                num_scenarios = params.num_mazes_train
            else:                
                num_scenarios = params.num_mazes_test
            
            chunk_size = num_scenarios // num_envs
            print('scenarios, chunk size')
            print(num_scenarios, chunk_size)
            
            chunks = chunk(range(num_scenarios), chunk_size)
            
            for idx, (in_queue, out_queue, idx_range) in enumerate(zip(self.in_queues, self.out_queues, chunks)):
                process = mp.Process(target=worker2, args=(in_queue, out_queue, params, is_train, idx_range), daemon=True) # use daemon=true so jobs die when there is an exception in main thread
                self.workers.append(process)
                print('starting process', idx)
                process.start()            
                time.sleep(params.sleep)
                gc.collect()
                print('sleep finished', idx)
            
        else:
            for idx, (in_queue, out_queue) in enumerate(zip(self.in_queues, self.out_queues)):
                process = mp.Process(target=worker, args=(in_queue, out_queue, params, is_train, idx), daemon=True) # use daemon=true so jobs die when there is an exception in main thread
                self.workers.append(process)
                process.start()
            
        print('There are {} workers'.format(len(self.workers)))
    
        assert env_id == 'doom', 'Multiprocessing only implemented for doom envirnment'           
        # tmp_env = DoomEnvironment(params)
        if params.num_actions == 0:
            num_actions = 5 if params.limit_actions else 8
            params.num_actions = num_actions
        self.num_actions = params.num_actions
        self.obs_shape = (3, params.screen_height, params.screen_width)
        self.prep = False # Observations already in CxHxW order    

    def reset(self):
        new_obs = []
        for queue in self.in_queues:
            queue.put('reset')
        for queue in self.out_queues:
            obs = queue.get()       
            new_obs.append(self.prep_obs(obs))
            
        return np.stack(new_obs)
    
    def get_depths(self, trim=True):
        depths = []
        command = 'depth'
        if trim: command = 'depth_trim'
            
        for queue in self.in_queues:
            queue.put(command)
        
        for queue in self.out_queues:
            depths.append(queue.get())

        return np.stack(depths)   
    
    def get_ego_depths(self, trim=False):
        ego_depths = []
        command = 'ego_depth'
        if trim: command = 'ego_depth_trim'    
        for queue in self.in_queues:
            queue.put(command)
        
        for queue in self.out_queues:
            ego_depths.append(queue.get())

        return np.stack(ego_depths)   
    
    def get_positions(self):
        positions = []
        command = 'positions'
            
        for queue in self.in_queues:
            queue.put(command)
        
        for queue in self.out_queues:
            x, y, theta = queue.get()
            positions.append([x, y, theta])

        return np.array(positions)         
    
    
    def get_deltas(self):
        deltas = []
        command = 'deltas'
            
        for queue in self.in_queues:
            queue.put(command)
        
        for queue in self.out_queues:
            dx, dy, dtheta = queue.get()
            deltas.append([dx, dy, dtheta])

        return np.array(deltas) 
    
    def get_origins(self):
        origins = []
        command = 'origins'
            
        for queue in self.in_queues:
            queue.put(command)
        
        for queue in self.out_queues:
            origin_x, origin_y = queue.get()
            origins.append([origin_x, origin_y])

        return np.array(origins)   
    
    def get_pos_deltas_origins(self):
        pos_delta_origins = []
        command = 'pos_deltas_origins'
            
        for queue in self.in_queues:
            queue.put(command)
        
        for queue in self.out_queues:
            x, y, theta, dx, dy, dtheta, origin_x, origin_y = queue.get()
            pos_delta_origins.append([x, y, theta, dx, dy, dtheta, origin_x, origin_y])

        return np.array(pos_delta_origins)          


    def cancel(self):
        for queue in self.in_queues:
            queue.put(None)     
            
        for worker in self.workers:
            worker.join()
        print('workers cancelled')
        
                    
    def prep_obs(self, obs):
        if self.prep:
            return obs.transpose(2,0,1)
        else: 
            return obs
    
    def step(self, actions):
        new_obs = []
        rewards = []
        dones = []
        infos = []
 
        for action, queue in zip(actions, self.in_queues):
            queue.put(action)
        
        for queue in self.out_queues:
            obs, reward, done, info = queue.get()
            new_obs.append(self.prep_obs(obs))
            rewards.append(reward)
            dones.append(done)
            infos.append(infos)
        
        return np.stack(new_obs), rewards, dones, infos                
       

class MultiEnvsMPPipes(object):
    """
        Run many envs on different processes to speed up simulation.
        
        Here this is fixed to be 16 workers but this could be increased if more
        compute is available
    """
    def __init__(self, env_id, num_envs, num_processes, params, is_train=True):
        
        self.parent_pipes, self.child_pipes = zip(*[mp.Pipe() for _ in range(num_envs)])
        
        self.workers = []
        
        if params.fixed_scenario:
            if is_train:
                num_scenarios = params.num_mazes_train
            else:                
                num_scenarios = params.num_mazes_test
            
            chunk_size = num_scenarios // num_envs
            print('scenarios, chunk size')
            print(num_scenarios, chunk_size)
            
            chunks = chunk(range(num_scenarios), chunk_size)
            
            for idx, (child_pipe, idx_range) in enumerate( zip(self.child_pipes, chunks)):
                process = mp.Process(target=pipe_worker2, args=(child_pipe, params, is_train, idx_range), daemon=True) # use daemon=true so jobs die when there is an exception in main thread
                self.workers.append(process)
                process.start()            
            
        else:
            for idx, child_pipe in enumerate( self.child_pipes):
                process = mp.Process(target=pipe_worker, args=(child_pipe, params, is_train, idx), daemon=True) # use daemon=true so jobs die when there is an exception in main thread
                self.workers.append(process)
                process.start()
            
        print('There are {} workers'.format(len(self.workers)))
    
        assert env_id == 'doom', 'Multiprocessing only implemented for doom envirnment'           
        # tmp_env = DoomEnvironment(params)
        if params.num_actions == 0:
            num_actions = 5 if params.limit_actions else 8
            params.num_actions = num_actions
        self.num_actions = params.num_actions
        
        if params.depth_as_obs:
            self.obs_shape = (4, params.screen_height, params.screen_width)
        else:
            self.obs_shape = (3, params.screen_height, params.screen_width)
        self.prep = False # Observations already in CxHxW order    

    def reset(self):
        new_obs = []
        for pipe in self.parent_pipes:
            pipe.send('reset')
            
        for pipe in self.parent_pipes:
            obs = pipe.recv()
            new_obs.append(self.prep_obs(obs))
            
        return np.stack(new_obs)
    
    def get_depths(self, trim=True):
        depths = []
        command = 'depth'
        if trim: command = 'depth_trim'
            
        for pipe in self.parent_pipes:
            pipe.send(command)
        
        for pipe in self.parent_pipes:
            depths.append(pipe.recv())

        return np.stack(depths)   
    
    def get_ego_depths(self, trim=False):
        ego_depths = []
        command = 'ego_depth'
        if trim: command = 'ego_depth_trim'    
        for pipe in self.parent_pipes:
            pipe.send(command)
        
        for pipe in self.parent_pipes:
            ego_depths.append(pipe.recv())

        return np.stack(ego_depths)   
    
    def get_positions(self):
        positions = []
        command = 'positions'
            
        for pipe in self.parent_pipes:
            pipe.send(command)
        
        for pipe in self.parent_pipes:
            x, y, theta = pipe.recv()
            positions.append([x, y, theta])

        return np.array(positions)         
    
    
    def get_deltas(self):
        deltas = []
        command = 'deltas'
            
        for pipe in self.parent_pipes:
            pipe.send(command)
        
        for pipe in self.parent_pipes:
            dx, dy, dtheta = pipe.recv()
            deltas.append([dx, dy, dtheta])

        return np.array(deltas) 
    
    def get_origins(self):
        origins = []
        command = 'origins'
            
        for pipe in self.parent_pipes:
            pipe.send(command)
        
        for pipe in self.parent_pipes:
            origin_x, origin_y = pipe.recv()
            origins.append([origin_x, origin_y])

        return np.array(origins)   
    
    def get_pos_deltas_origins(self):
        pos_delta_origins = []
        command = 'pos_deltas_origins'
            
        for pipe in self.parent_pipes:
            pipe.send(command)
        
        for pipe in self.parent_pipes:
            x, y, theta, dx, dy, dtheta, origin_x, origin_y = pipe.recv()
            pos_delta_origins.append([x, y, theta, dx, dy, dtheta, origin_x, origin_y])

        return np.array(pos_delta_origins)  
        
    def get_loops(self):
        loops = []
        command = 'loops'
            
        for pipe in self.parent_pipes:
            pipe.send(command)
        
        for pipe in self.parent_pipes:
            loop = pipe.recv()
            loops.append([loop])

        return np.array(loops)          


    def cancel(self):
        for pipe in self.parent_pipes:
            pipe.send(None)     
            
        for worker in self.workers:
            worker.join()
        print('workers cancelled')
        
                    
    def prep_obs(self, obs):
        if self.prep:
            return obs.transpose(2,0,1)
        else: 
            return obs
    
    def step(self, actions):
        new_obs = []
        rewards = []
        dones = []
        infos = []
 
        for action, pipe in zip(actions, self.parent_pipes):
            pipe.send(action)
        
        for pipe in self.parent_pipes:
            obs, reward, done, info = pipe.recv()
            new_obs.append(self.prep_obs(obs))
            rewards.append(reward)
            dones.append(done)
            infos.append(infos)
        
        return np.stack(new_obs), rewards, dones, infos                
       
        
class MultiEnvs(object):
    
    def __init__(self, env_id, num_envs, num_processes, params):  
        if env_id == 'doom':
            # for the doom scenarios
            self.envs = [DoomEnvironment(params) for i in range(num_envs)]
            self.num_actions = self.envs[0].num_actions
            self.obs_shape = (3, params.screen_height, params.screen_width)
            self.prep = False # Observations already in CxHxW order
        elif env_id == 'home':
            assert 0, 'HoME has not been implemented yet'
        else:
            # if testing on Atari games such as Pong etc
            self.envs = [wrap_deepmind(make_atari(env_id)) for i in range(num_envs)]
            observation_space = self.envs[0].observation_space
            obs_shape = observation_space.shape
            observation_space = Box(
                observation_space.low[0,0,0],
                observation_space.high[0,0,0],
                [obs_shape[2], obs_shape[1], obs_shape[0]]
            )
            action_space  = self.envs[0].action_space
        
            self.num_actions = action_space.n
            self.obs_shape = observation_space.shape
            self.prep = True
            
        
    
    def reset(self):
        return np.stack([self.prep_obs(env.reset()) for env in self.envs])
    
    def get_depths(self, trim=True):
        if trim:
            return np.stack([env.get_depth()[2:-2,2:-2] for env in self.envs])
        else:
            return np.stack([env.get_depth() for env in self.envs])
    
    def prep_obs(self, obs):
        if self.prep:
            return obs.transpose(2,0,1)
        else: 
            return obs
    
    def step(self, actions):
        new_obs = []
        rewards = []
        dones = []
        infos = []
        
        
        for env, action in zip(self.envs, actions):
            obs, reward, done, info = env.step(action)
#            if done: 
#                obs = env.reset()
            new_obs.append(self.prep_obs(obs))
            rewards.append(reward)
            dones.append(done)
            infos.append(infos)
        
        return np.stack(new_obs), rewards, dones, infos
    

if __name__ == '__main__':
    
    params = parse_game_args()
    params.scenario_dir = '../resources/scenarios/'
    
    # env = DoomEnvironment(params, idx=0, is_train=True, use_shaping=params.use_shaping)
    
    # obs, reward, done, info = env.step(0)
    

    mp_test_envs = MultiEnvsMP(params.simulator, params.num_environments, 1, params)
    mp_test_envs.reset()
    

    
    actions = [2]*16
    
    # for i in range(10):
    #     new_obs, rewards, dones, infos = mp_test_envs.step(actions)
    #     print(rewards, np.stack(rewards))    
    
    envs = MultiEnvs(params.simulator, params.num_environments, 1, params)
    envs.reset()
    


    def test_mp_reset():
        mp_test_envs.reset()
    
    def test_mp_get_obs():
        actions = [2]*16
        new_obs, rewards, dones, infos = mp_test_envs.step(actions)
    
    def test_sp_reset():
        envs.reset()
    
    def test_sp_get_obs():
        actions = [2]*16
        new_obs, rewards, dones, infos = envs.step(actions)        


    











