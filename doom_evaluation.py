#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 14:31:17 2018

@author: anonymous
"""
if __name__ == '__main__': #changes backend for animation tests
    import matplotlib 
    matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from moviepy.editor import ImageSequenceClip

from environments import DoomEnvironment

import torch
from torch import Tensor
from torch.autograd import Variable
from doom_a2c.arguments import parse_game_args
from doom_a2c.multi_env import MultiEnvs
from doom_a2c.models import CNNPolicy

class BaseAgent(object):
    def __init__(self, model, params):
        self.params = params
        self.model = model
        self.ego_model = params.ego_model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if params.num_stack > 1:
            self.exp_size = params.num_stack
            self.short_term_memory = deque()

        if params.learn_init_state:
            self.state = model.init_state.clone().to(self.device)
        else:
            self.state = torch.zeros(1, model.state_size).to(self.device)
            
        if params.use_lstm:
            self.cells = torch.zeros(1, model.state_size).to(self.device)
        else:
            self.cells = None
        
        if params.ego_model:
            self.ego_state = torch.zeros(1, params.ego_num_chans, params.ego_half_size*2, params.ego_half_size*2).to(self.device)
        else: 
            self.ego_state = None

        self.mask = Tensor([[1.0]]).to(self.device)
        
                
    def get_salency(self, observation):
        # TODO
   
        forward_outputs  = []
        backward_outputs  = []    
        
        def forward_hook(self, input, output):
            forward_outputs.append(output[0].clone())
            
            
        def backward_hook(self, grad_input, grad_output):
            backward_outputs.append(grad_output[0].clone())

        self.model.conv_head[5].register_forward_hook(forward_hook)
        self.model.conv_head[5].register_forward_hook(backward_hook)
        
        observation = torch.from_numpy(observation).unsqueeze(0) 
        
    def get_action(self, observation, epsilon=0.0, ego_depth=None, pos_deltas_origins=None):
        if hasattr(self, 'short_term_memory'):
            observation = self._prepare_observation(observation)    
        observation = torch.from_numpy(observation).unsqueeze(0).to(self.device) 
        _, action, _, self.state, self.cells, self.ego_state = self.model.act(observation, self.state, self.mask, 
                                                                              deterministic=True, cells=self.cells, pos_deltas_origins=pos_deltas_origins,
                                                                              ego_states=self.ego_state, ego_depths=ego_depth)     

        return action.cpu().data.numpy()[0,0]
        


    def get_action_value_and_probs(self, observation, epsilon=0.0, deterministic=True, ego_depth=None, pos_deltas_origins=None):
        if hasattr(self, 'short_term_memory'):
            observation = self._prepare_observation(observation)
        
        observation = torch.from_numpy(observation).unsqueeze(0).to(self.device)

        result  = self.model.get_action_value_and_probs(observation, self.state, self.mask,
                                                        deterministic=deterministic,
                                                        ego_states=self.ego_state, ego_depths=ego_depth,
                                                        pos_deltas_origins=pos_deltas_origins)
        
        value = result['values']
        action = result['actions']
        probs = result['action_softmax']
        self.state = result['states']
        self.ego_state = result.get('ego_states', None)
        
        return action.cpu().detach().numpy()[0,0], value.cpu().detach().numpy(), probs.cpu().detach().numpy()
        
    def reset(self):
        """
            reset the models hidden layer when starting a new rollout
        """
        if hasattr(self, 'short_term_memory'):
            self.short_term_memory = deque()
            
        if self.params.learn_init_state:
            self.state = self.model.init_state.clone().to(self.device)
        else:
            self.state = torch.zeros(1, self.model.state_size).to(self.device)
            
        if self.ego_state is not None:
            self.ego_state = self.ego_state * 0      
        
     
    def _prepare_observation(self, observation):
        """
           As the network expects an input of n frames, we must store a small
           short term memory of frames. At input this is completely empty so 
           I pad with the firt observations 4 times, generally this is only used when the network
           is not recurrent
        """
        if len(self.short_term_memory) == 0 :
            for _ in range(self.exp_size):
                self.short_term_memory.append(observation)
            
        self.short_term_memory.popleft()
        self.short_term_memory.append(observation)
        
        return np.vstack(self.short_term_memory)
    
       
def eval_model(model, params, logger, step, train_iters, num_games, movie=True, is_train=False):
    agent = BaseAgent(model, params)
    
    print('agent created')
    if params.multimaze:
        accumulated_rewards = 0
        for env_idx in range(params.num_environments):
            env = DoomEnvironment(params, idx=env_idx, is_train=is_train)
            mean_reward = eval_agent_multi(agent, env, logger, params, step, train_iters, num_games=10, env_idx=env_idx, movie=movie)
            accumulated_rewards += mean_reward
        logger.write('Step: {:0004}, Iter: {:000000008} Eval mean reward: {:0003.3f}'.format(step, train_iters, accumulated_rewards / params.num_environments))
    else:        
        env = DoomEnvironment(params)
        print('eval agent')
        eval_agent(agent, env, logger, params, step, train_iters, num_games, movie=movie)

    
def eval_agent_multi(agent, env, logger, params,  step, train_iters, num_games=10, env_idx=0, movie=True):
    """
        Evaluates an agents performance in an environment Two metrics are
        computed: number of games suceeded and average total reward.
    """
    # TODO: Back up the enviroment so the agent can start where it left off    
    best_obs = None
    worst_obs = None
    best_reward = -100000
    worst_reward = 100000
    accumulated_rewards = 0.0
    reward_list = []
    time_list = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for game in range(num_games):
        env.reset()
        agent.reset()
        k = 0
        rewards = []
        obss = []
        while not env.is_episode_finished():
            obs = env.get_observation()

            pos_deltas_origins, ego_depth  = None, None
            if params.ego_model:
                pos_deltas_origins = env.get_player_pos_delta_origin()
                pos_deltas_origins = torch.from_numpy(np.array(pos_deltas_origins).unsqueeze(0).float()).to(device)

                ego_depth = torch.from_numpy(env.get_ego_depth()).unsqueeze(0).to(device)
                 
            action = agent.get_action(obs, epsilon=0.0, ego_depth=ego_depth, 
                                      pos_deltas_origins=pos_deltas_origins)
            
            reward = env.make_action(action)
            rewards.append(reward)
            if not params.norm_obs:
                obs = obs*(1.0/255.0)
            obss.append(obs)
            k += 1
        time_list.append(k)
        
        reward_list.append(env.get_total_reward())
        if env.get_total_reward() > best_reward:
            best_obs = obss
            best_reward = env.get_total_reward()
        if env.get_total_reward() < worst_reward:
            worst_obs = obss
            worst_reward = env.get_total_reward()              
  
        accumulated_rewards += env.get_total_reward()
        
    if movie:
        write_movie(params, logger, best_obs, step + 2*env_idx, best_reward, best_agent=True)
        write_movie(params, logger, worst_obs, step+1 + 2*env_idx, worst_reward, best_agent=False)    

    logger.write('Step: {:0004}, env_id:{} Game rewards: {}, Game times: {}'.format(step, env_idx, reward_list, time_list))
    
    return accumulated_rewards / num_games
       
    
def eval_agent(agent, env, logger, params,  step, train_iters, num_games=10, movie=True):
    """
        Evaluates an agents performance in an environment Two metrics are
        computed: number of games suceeded and average total reward.
    """
    # TODO: Back up the enviroment so the agent can start where it left off    
    best_obs = None
    worst_obs = None
    best_reward = -10000
    worst_reward = 100000
    accumulated_rewards = 0.0
    reward_list = []
    time_list = []
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for game in range(num_games):
        
        env.reset()
        agent.reset()
                
        k = 0
        rewards = []
        obss = []
        
        while not env.is_episode_finished():
            obs = env.get_observation()

            pos_deltas_origins, ego_depth  = None, None
            if params.ego_model:

                pos_deltas_origins = env.get_player_pos_delta_origin()
                pos_deltas_origins = torch.from_numpy(np.array(pos_deltas_origins)).unsqueeze(0).float().to(device)
                ego_depth = env.get_ego_depth()
                if not params.new_padding:
                    ego_depth = ego_depth[2:-2,2:-2]
                ego_depth = torch.from_numpy(ego_depth).unsqueeze(0).to(device)
      
            action = agent.get_action(obs, epsilon=0.0, ego_depth=ego_depth, 
                                      pos_deltas_origins=pos_deltas_origins)
 
            reward = env.make_action(action)
            rewards.append(reward)
                        
            if not params.norm_obs:
                obs = obs*(1.0/255.0)
            obss.append(obs)
            k += 1
        time_list.append(k)

        reward_list.append(env.get_total_reward())
        if env.get_total_reward() > best_reward:
            best_obs = obss
            best_reward = env.get_total_reward()
        if env.get_total_reward() < worst_reward:
            worst_obs = obss
            worst_reward = env.get_total_reward()              
  
        accumulated_rewards += env.get_total_reward()
        
    if params.use_visdom:
        logger.vis_iters.append(train_iters)
        logger.vis_scores.append(accumulated_rewards / num_games)
        logger.update_plot(train_iters)
        
    if movie:
        write_movie(params, logger, best_obs, step, best_reward)
        write_movie(params, logger, worst_obs, step+1, worst_reward, best_agent=False)    
        
    logger.write('Step: {:0004}, Iter: {:000000008} Eval mean reward: {:0003.3f}'.format(step, train_iters, accumulated_rewards / num_games))
    logger.write('Step: {:0004}, Game rewards: {}, Game times: {}'.format(step, reward_list, time_list))


        


def write_movie(params,logger,  observations, step, score, best_agent=True):    
    observations = [o.transpose(1,2,0)*255.0 for o in observations]
    clip = ImageSequenceClip(observations, fps=int(30/params.frame_skip))
    output_dir = logger.get_eval_output()
    clip.write_videofile('{}eval{:0004}_{:00005.0f}.mp4'.format(output_dir, step, score*100))  
    if params.use_visdom:
        logger.add_video('{}eval{:0004}_{:00005.0f}.mp4'.format(output_dir, step, score*100), best_agent=best_agent)
    
    
  
if __name__ == '__main__':
    # Test to improve movie with action probs, values etc
    
    params = parse_game_args()    
    params.norm_obs = False
    params.recurrent_policy = True
    envs = MultiEnvs(params.simulator, 1, 1, params)
    obs_shape = envs.obs_shape
    obs_shape = (obs_shape[0] * params.num_stack, *obs_shape[1:])    
    model = CNNPolicy(obs_shape[0], envs.num_actions, params.recurrent_policy, obs_shape)
    env = DoomEnvironment(params)
    agent = BaseAgent(model, params)
    
    env.reset()
    agent.reset()
    
    rewards = []
    obss = []  
    actions = []
    action_probss = []
    values = []
    
    while not env.is_episode_finished():
        obs = env.get_observation()
        #action = agent.get_action(obs, epsilon=0.0)
        action, value, action_probs = agent.get_action_value_and_probs(obs, epsilon=0.0)
        #print(action)
        reward = env.make_action(action)
        rewards.append(reward)
        obss.append(obs)    
        actions.append(actions)
        action_probss.append(action_probs)
        values.append(value)
    
    
    value_queue = deque()
    reward_queue = deque()
    for i in range(64):    
        value_queue.append(0.0)
        reward_queue.append(0.0)
    

    import matplotlib.animation as manimation
    
    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='Movie Test', artist='Edward Beeching',
                    comment='First movie with data')
    writer = FFMpegWriter(fps=7.5, metadata=metadata)    

    
    #plt.style.use('seaborn-paper')
    fig = plt.figure(figsize=(16,9))
    
    ax1 = plt.subplot2grid((6,6), (0,0), colspan=6, rowspan=4)
    ax2 = plt.subplot2grid((6,6), (4,3), colspan=3, rowspan=2)
    ax3 = plt.subplot2grid((6,6), (4,0), colspan=3, rowspan=1)
    ax4 = plt.subplot2grid((6,6), (5,0), colspan=3, rowspan=1)
    # World plot
    im = ax1.imshow(obs.transpose(1,2,0)/255.0)
    ax1.axis('off')
    
    # Action plot
    bar_object = ax2.bar('L, R, F, B, L + F, L + B, R + F, R + B'.split(','), action_probs.tolist()[0]) 
    ax2.set_title('Action Probabilities', position=(0.5, 0.85))
    
    #plt.title('Action probabilities')
    #ax2.axis('on')
    ax2.set_ylim([-0.01, 1.01])
    # values
    values_ob, = ax3.plot(value_queue) 
    ax3.set_title('State Values', position=(0.1,0.05))
    ax3.set_ylim([np.min(np.stack(values))-0.2,np.max(np.stack(values))+0.2]) 
    ax3.get_xaxis().set_visible(False)
    #plt.title('State values')
    rewards_ob, = ax4.plot(reward_queue) 
    ax4.set_title('Rewards', position=(0.07, 0.05))
    #plt.title('Reward values')
    ax4.set_ylim([-0.01, 1.0])    
    fig.tight_layout()
    
    print('writing')
    with writer.saving(fig, "writer_test.mp4", 100):
        for observation, action_probs, value, reward in zip(obss, action_probss, values, rewards):
            im.set_array(observation.transpose(1,2,0)/255.0)    
            for b, v in zip(bar_object, action_probs.tolist()[0]):
                b.set_height(v)
            value_queue.popleft()
            value_queue.append(value[0,0])
            reward_queue.popleft()
            reward_queue.append(reward)
            values_ob.set_ydata(value_queue)
            rewards_ob.set_ydata(reward_queue)
            
            writer.grab_frame()
    
       
    
