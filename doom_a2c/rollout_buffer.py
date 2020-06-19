#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 09:13:08 2018

@author: anonymous

A buffer for holding rollouts from multiple simulators

Built upon implementation found at https://github.com/ikostrikov/pytorch-a2c-ppo-acktr

"""
import torch


class RolloutStorage(object):
    def __init__(self, num_steps, num_processes, obs_shape, state_size, params):
        
        self.observations = torch.zeros(num_steps + 1, num_processes, *obs_shape)
        self.states = torch.zeros(num_steps + 1, num_processes, state_size)
        self.rewards = torch.zeros(num_steps, num_processes, 1)
        self.value_preds = torch.zeros(num_steps + 1, num_processes, 1)
        self.returns = torch.zeros(num_steps + 1, num_processes, 1)
        self.action_log_probs = torch.zeros(num_steps, num_processes, 1)
        self.actions = torch.zeros(num_steps, num_processes, 1).long()
        self.masks = torch.zeros(num_steps + 1, num_processes, 1)

        
        if params.ego_curiousity:
            self.previous_observations = torch.zeros(num_steps + 1, num_processes, *obs_shape)
            self.previous_actions = torch.zeros(num_steps +1, num_processes, 1).long()
            self.curiousities = torch.zeros(num_steps + 1, num_processes, 1, 4, 10) # TODO resize to cnn output
        
        if params.loop_detect:
            self.loops = torch.zeros(num_steps + 1, num_processes, 1)

        if params.stacked_gru:
            self.states2 = torch.zeros(num_steps + 1, num_processes, state_size)

        if params.predict_depth:
            self.depths = torch.zeros(num_steps + 1, num_processes, 4, 10)
            self.depths = self.depths.long()
        
        if params.use_lstm:
            self.cells = torch.zeros(num_steps + 1, num_processes, state_size)
            
        if params.neural_map:
             self.ego_states = torch.zeros(num_steps + 1, 
                                          num_processes, 
                                          params.ego_num_chans, 
                                          params.ego_half_size*2 - 1, 
                                          params.ego_half_size*2 - 1) # this map has the agent at the exact centre          
             self.pos_deltas_origins = torch.zeros(num_steps + 1, num_processes, 8)
             
        if params.ego_model:
            ac = 0
            if params.ego_curiousity:
                ac = 1
            
            self.ego_states = torch.zeros(num_steps + 1, 
                                          num_processes, 
                                          params.ego_num_chans + ac, 
                                          params.ego_half_size*2, 
                                          params.ego_half_size*2)
            
            subtract = 0
            if not params.new_padding:
                subtract = 4

            self.ego_depths = torch.zeros(num_steps + 1, num_processes, 
                                          params.screen_height // 8 - subtract,
                                          params.screen_width // 8 - subtract)
            print(self.ego_depths.size())
            
            self.pos_deltas_origins = torch.zeros(num_steps + 1, num_processes, 8)
        
        if params.pos_as_obs:
            self.pos_deltas_origins = torch.zeros(num_steps + 1, num_processes, 8)
        



    def set_device(self, device):
        self.observations = self.observations.to(device)
        self.states = self.states.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)
        
        if hasattr(self, 'curiousities'):
            self.curiousities = self.curiousities.to(device) 
            
        if hasattr(self, 'previous_observations'):
            self.previous_observations = self.previous_observations.to(device) 
            
        if hasattr(self, 'previous_actions'):
            self.previous_actions = self.previous_actions.to(device) 
            
        if hasattr(self, 'loops'):
            self.loops = self.loops.to(device)   
        if hasattr(self, 'states2'):
            self.states2 = self.states2.to(device)   
        if hasattr(self, 'depths'):
            self.depths = self.depths.to(device)   
        if hasattr(self, 'cells'):
            self.cells = self.cells.to(device)

        if hasattr(self, 'ego_states'):
            self.ego_states = self.ego_states.to(device)
        if hasattr(self, 'ego_depths'):
            self.ego_depths = self.ego_depths.to(device)
        if hasattr(self, 'pos_deltas_origins'):
            self.pos_deltas_origins = self.pos_deltas_origins.to(device)
          

    def insert(self, step, current_obs, state, action, 
               action_log_prob, value_pred, reward, mask, 
               depths=None, cells=None, 
               ego_states=None, ego_depths=None, pos_deltas_origins=None,
               loops=None, states2=None, prev_obs=None, curiousity=None): 
        
        self.observations[step + 1].copy_(current_obs)
        self.states[step + 1].copy_(state)
        
        if states2 is not None and hasattr(self, 'states2'):
            self.states2[step + 1].copy_(states2)
        
        self.actions[step].copy_(action)
        self.action_log_probs[step].copy_(action_log_prob)
        self.value_preds[step].copy_(value_pred)
        self.rewards[step].copy_(reward)
        self.masks[step + 1].copy_(mask)
        
        if prev_obs is not None and hasattr(self, 'previous_observations'):
            # TODO wrong
            self.previous_observations[step + 1].copy_(prev_obs)
            self.curiousities[step + 1].copy_(curiousity)
            self.previous_actions[step + 1].copy_(action)

        if loops is not None and hasattr(self, 'loops'):
            self.loops[step + 1].copy_(loops)
        
        if depths is not None and hasattr(self, 'depths'):
            self.depths[step + 1].copy_(depths)
        
        if cells is not None and hasattr(self, 'cells'):
            self.cells[step + 1].copy_(cells)
        
        if ego_states is not None and hasattr(self, 'ego_states'):
            self.ego_states[step + 1].copy_(ego_states)
            
        if ego_depths is not None and hasattr(self, 'ego_depths'):
            self.ego_depths[step + 1].copy_(ego_depths)
            
        if pos_deltas_origins is not None and hasattr(self, 'pos_deltas_origins'):
            self.pos_deltas_origins[step + 1].copy_(pos_deltas_origins)
            
            
    def after_update(self):
        self.observations[0].copy_(self.observations[-1])
        self.states[0].copy_(self.states[-1])
        self.masks[0].copy_(self.masks[-1])
        
        if hasattr(self, 'previous_actions'):
            self.previous_actions[0].copy_(self.previous_actions[-1])
        if hasattr(self, 'previous_observations'):
            self.previous_observations[0].copy_(self.previous_observations[-1])
            
        if hasattr(self, 'curiousities'):
            self.curiousities[0].copy_(self.curiousities[-1])
            
        if hasattr(self, 'loops'):
            self.loops[0].copy_(self.loops[-1])
        
        if hasattr(self, 'states2'):
            self.states2[0].copy_(self.states2[-1])
        
        if hasattr(self, 'depths'):
            self.depths[0].copy_(self.depths[-1])
            
        if hasattr(self, 'cells'):
            self.cells[0].copy_(self.cells[-1])
            
        if hasattr(self, 'ego_states'):
            self.ego_states[0].copy_(self.ego_states[-1])
            
        if hasattr(self, 'ego_depths'):
            self.ego_depths[0].copy_(self.ego_depths[-1])
            
        if hasattr(self, 'pos_deltas_origins'):
            self.pos_deltas_origins[0].copy_(self.pos_deltas_origins[-1])
            



    def compute_returns(self, next_value, use_gae, gamma, tau):
        if use_gae:
            self.value_preds[-1] = next_value
            gae = 0
            for step in reversed(range(self.rewards.size(0))):
                delta = self.rewards[step] + gamma * self.value_preds[step + 1] * self.masks[step + 1] - self.value_preds[step]
                gae = delta + gamma * tau * self.masks[step + 1] * gae
                self.returns[step] = gae + self.value_preds[step]
        else:
            self.returns[-1] = next_value
            for step in reversed(range(self.rewards.size(0))):
                self.returns[step] = self.returns[step + 1] * \
                    gamma * self.masks[step + 1] + self.rewards[step]
                    
                   
