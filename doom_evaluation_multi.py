#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 08/10/2018

@author: anonymous
"""



if __name__ == '__main__': #changes backend for animation tests
    import matplotlib 
    matplotlib.use("Agg")
import numpy as np
from moviepy.editor import ImageSequenceClip
import gc
import torch
from doom_a2c.arguments import parse_game_args
from doom_a2c.multi_env import MultiEnvs
from doom_a2c.models import CNNPolicy
from doom_a2c.multi_env import MultiEnvsMP, MultiEnvsMPPipes



class Scorer():
    def __init__(self, num_envs, initial_obs, movie=True):
        self.best = [None, -100000] # obs, and best reward
        self.worst = [None, 100000] # obs, and worse reward
        self.trajectories = {}
        self.total_rewards = []
        self.total_times = []
        self.num_envs = num_envs
        self.movie = movie
        if self.movie:
            initial_obs = initial_obs.astype(np.uint8)           
        else:
            initial_obs = [None]*initial_obs.shape[0]
        
        for i in range(num_envs):
            self.trajectories[i] = [[initial_obs[i]], []]
            
    def update(self, obs, rewards, dones):   
        obs = obs.astype(np.uint8)   
        if self.movie:
            obs = obs.astype(np.uint8)           
        else:
            obs = [None]*obs.shape[0]
        
        
        for i in range(self.num_envs):
            if dones[i]:
                self.trajectories[i][1].append(rewards[i])
                
                accumulated_reward = sum(self.trajectories[i][1])
                
                self.total_rewards.append(accumulated_reward)
                self.total_times.append(len(self.trajectories[i][1]))
                
                if accumulated_reward > self.best[1]:
                    self.best[0] = self.trajectories[i][0]
                    self.best[1] = accumulated_reward
                    
                if accumulated_reward < self.worst[1]:
                    self.worst[0] = self.trajectories[i][0]
                    self.worst[1] = accumulated_reward   
             
                self.trajectories[i] = [[obs[i]], [0.0]]
  
            else:
                self.trajectories[i][0].append(obs[i])
                self.trajectories[i][1].append(rewards[i])
                
                
    def clear(self):
        self.trajectories = None


def mem_report():
    for obj in gc.get_objects(): 
        if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)): 
            print(type(obj), obj.size())
       
        
class Evaluator():
    def __init__(self, params, is_train=False):
        self.num_envs = params.num_environments
        self.envs = MultiEnvsMPPipes(params.simulator, self.num_envs, 1, params, is_train=is_train)  
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
    
        self.states = torch.zeros(self.num_envs, params.hidden_size).to(self.device)
        
        if params.stacked_gru:
            self.states2 = torch.zeros(self.num_envs, params.hidden_size).to(self.device)
        else:
            self.states2 = None

        self.ego_depths = None
        self.pos_deltas_origins = None            
        if params.ego_model:
            ac = 0
            if params.ego_curiousity:
                ac = 1
            
            self.ego_states = torch.zeros(self.num_envs, 
                                    params.ego_num_chans + ac, 
                                    params.ego_half_size*2, 
                                    params.ego_half_size*2).to(self.device)
        elif params.neural_map:
            self.ego_states = torch.zeros(self.num_envs, 
                                    params.ego_num_chans, 
                                    params.ego_half_size*2 -1, 
                                    params.ego_half_size*2 -1).to(self.device)
        else: 
            self.ego_states = None
        
        
    def cancel(self):
        self.envs.cancel()
    
    def evaluate(self, model,  params,logger,  step, train_iters, num_games=10, movie=True):
        model.eval()
        
        games_played = 0
        obs = self.envs.reset()
        # add obs to scorer
        scorer = Scorer(self.num_envs, obs, movie=movie)
        obs = torch.from_numpy(obs).float().to(self.device)
        
        masks = torch.ones(self.num_envs, 1).to(self.device)

        self.states.zero_().detach()
        if params.stacked_gru:
            self.states2.zero_().detach()
            
          
        if params.ego_model:
            self.ego_states.zero_().detach()
        elif params.neural_map:
            self.ego_states.zero_().detach()

        if params.ego_curiousity:
            prev_obs = obs.clone()
            prev_actions = torch.zeros(16,1).long().to(self.device)


        while games_played < num_games:
            ego_depths = None
            pos_deltas_origins = None
            if params.neural_map:
                pos_deltas_origins = self.envs.get_pos_deltas_origins()
                pos_deltas_origins = torch.from_numpy(np.array(pos_deltas_origins)).float().to(self.device)               
            
            if params.ego_model:
                pos_deltas_origins = self.envs.get_pos_deltas_origins()
                pos_deltas_origins = torch.from_numpy(np.array(pos_deltas_origins)).float().to(self.device)
                ego_depths = torch.from_numpy(self.envs.get_ego_depths(trim=(not params.new_padding))).to(self.device)
                
            if params.pos_as_obs:
                pos_deltas_origins = self.envs.get_pos_deltas_origins()
                pos_deltas_origins = torch.from_numpy(np.array(pos_deltas_origins)).float().to(self.device)                
                
            kwargs = {'ego_states': self.ego_states, 
                      'states2': self.states2,
                      'ego_depths': ego_depths, 
                      'pos_deltas_origins': pos_deltas_origins}
            
            if params.ego_curiousity:
                kwargs['prev_obs'] = prev_obs
                kwargs['prev_actions'] = prev_actions
            
            result = model.act(obs, self.states.detach(), masks,
                               deterministic=not params.stoc_eval, **kwargs)        
    
            actions = result['actions']
            prev_actions = actions
            self.states = result['states']
            self.states2 = result.get('states2', None)
            self.ego_states = result.get('ego_states', None)
    
            cpu_actions = actions.squeeze(1).cpu().numpy()
            if params.ego_curiousity:
                prev_obs = obs.clone()
            
            obs, reward, done, info = self.envs.step(cpu_actions)
            # add obs, reward,  to scorer
            scorer.update(obs, reward, done)
    
            games_played += done.count(True) # done is true at end of a turn

            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            masks = masks.to(self.device)
            obs = torch.from_numpy(obs).float().to(self.device)
    
        model.train()
        
         # it is possible that this is larger that the total num games
        accumulated_rewards = sum(scorer.total_rewards[:num_games])
        best_obs, best_reward = scorer.best
        worst_obs, worst_reward = scorer.worst
        reward_list = scorer.total_rewards[:num_games]
        time_list = scorer.total_times[:num_games]
        scorer.clear()
        
        if params.use_visdom:
            logger.vis_iters.append(train_iters)
            logger.vis_scores.append(accumulated_rewards / num_games)
            logger.update_plot(train_iters)
            
        if movie:
            write_movie(params, logger, best_obs, step, best_reward)
            write_movie(params, logger, worst_obs, step+1, worst_reward, best_agent=False)    
            
        logger.write('Step: {:0004}, Iter: {:000000008} Eval mean reward: {:0003.3f}'.format(step, train_iters, accumulated_rewards / num_games))
        logger.write('Step: {:0004}, Game rewards: {}, Game times: {}'.format(step, reward_list, time_list))        


def eval_model_multi(model,  params,logger,  step, train_iters, num_games=10, movie=True):
    """
        evaluate model using multiprocessing
    
    """
    num_envs = params.num_environments
    num_envs = 8
    envs = MultiEnvsMP(params.simulator, num_envs, 1, params)  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    games_played = 0
    obs = envs.reset()
    # add obs to scorer
    scorer = Scorer(num_envs, obs.copy())
    obs = torch.from_numpy(obs).float().to(device)
    masks = torch.ones(num_envs, 1).to(device)
    if params.learn_init_state:
        states = model.init_state.clone().repeat(num_envs, 1).to(device)
    else:
        states = torch.zeros(num_envs, model.state_size).to(device)

    if params.use_lstm:
        cells = torch.zeros(num_envs, model.state_size).to(device)
    else:
        cells = None
        
    if params.ego_model:
        ego_states = torch.zeros(num_envs, 
                                params.ego_num_chans, 
                                params.ego_half_size*2, 
                                params.ego_half_size*2).to(device)
    else: 
        ego_states = None
        ego_depths = None
        pos_deltas_origins = None
        
    while games_played < num_games:  
        if params.ego_model:
            pos_deltas_origins = envs.get_pos_deltas_origins()
            pos_deltas_origins = torch.from_numpy(np.array(pos_deltas_origins).float()).to(device)

            ego_depths = torch.from_numpy(envs.get_ego_depths()).to(device)
            
        kwargs = {'cells':cells, 
                  'ego_states': ego_states, 
                  'ego_depths':ego_depths, 
                  'pos_deltas_origins':pos_deltas_origins}
        
        _, actions, _, states, cells, ego_states = model.act(obs,
                                                             states,
                                                             masks,
                                                             **kwargs)        

        cpu_actions = actions.squeeze(1).cpu().numpy()
        obs, reward, done, info = envs.step(cpu_actions)
        # add obs, reward,  to scorer
        scorer.update(obs.copy(), reward, done)

        games_played += done.count(True) # done is true at end of a turn
        
        masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
        masks = masks.to(device)
        obs = torch.from_numpy(obs).float().to(device)

    envs.cancel()
    accumulated_rewards = sum(scorer.total_rewards[:num_games]) # it is possible that this is larger that the total num games
    best_obs, best_reward = scorer.best
    worst_obs, worst_reward = scorer.worst
    reward_list = scorer.total_rewards[:num_games]
    time_list = scorer.total_times[:num_games]
    scorer.clear()
    del scorer
    
    if params.use_visdom:
        logger.vis_iters.append(train_iters)
        logger.vis_scores.append(accumulated_rewards / num_games)
        logger.update_plot(train_iters)
        
    if movie:
        write_movie(params, logger, best_obs, step, best_reward)
        write_movie(params, logger, worst_obs, step+1, worst_reward, best_agent=False)    
        
    logger.write('Step: {:0004}, Iter: {:000000008} Eval mean reward: {:0003.3f}'.format(step, train_iters, accumulated_rewards / num_games))
    logger.write('Step: {:0004}, Game rewards: {}, Game times: {}'.format(step, reward_list, time_list))


def write_movie(params, logger,  observations, step, score, best_agent=True):    
    observations = [o.transpose(1,2,0) for o in observations]
    clip = ImageSequenceClip(observations, fps=int(30/params.frame_skip))
    output_dir = logger.get_eval_output()
    clip.write_videofile('{}eval{:0004}_{:00005.0f}.mp4'.format(output_dir, step, score*100))  
    if params.use_visdom:
        logger.add_video('{}eval{:0004}_{:00005.0f}.mp4'.format(output_dir, step, score*100), best_agent=best_agent)
    
      
def evalu(model, a, params, b, c, num_games=50):
    with torch.no_grad():
        eval_model_multi(model, a, params, b, c, num_games=50)
  
if __name__ == '__main__':
    params = parse_game_args()    
    params.norm_obs = False
    params.num_stack = 1
    params.recurrent_policy = True
    params.num_environments = 16
    params.scenario = 'scenario_3_item0.cfg'
    
    envs = MultiEnvs(params.simulator, 1, 1, params)
    obs_shape = envs.obs_shape
    obs_shape = (obs_shape[0] * params.num_stack, *obs_shape[1:])    
    model = CNNPolicy(obs_shape[0], obs_shape, params)

    with torch.no_grad():
        eval_model_multi(model,  params, 0, 0, 0, num_games=1000)    
        
        
  
