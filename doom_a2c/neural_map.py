#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 13:50:02 2018

@author: anonymous
"""

import math

import torch
from torch import nn
import torch.nn.functional as F
from .models import FFPolicy, orthogonal, Lin_View, weights_init

from .distributions import Categorical
            
class NeuralMapPolicy(FFPolicy):
    def __init__(self, num_inputs, input_shape, params):
        super(NeuralMapPolicy, self).__init__()
        
        self.params = params
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if params.new_padding:
            if params.skip_cnn_relu:
                self.conv_head = nn.Sequential(nn.Conv2d(num_inputs, params.conv1_size, 8, stride=4, padding=2),
                                            nn.ReLU(True),
                                            nn.Conv2d(params.conv1_size, params.conv2_size, 4, stride=2, padding=1),
                                            nn.ReLU(True),
                                            nn.Conv2d(params.conv2_size, params.ego_num_chans, 3, stride=1, padding=1))
            else:
                self.conv_head = nn.Sequential(nn.Conv2d(num_inputs, params.conv1_size, 8, stride=4, padding=2),
                                            nn.ReLU(True),
                                            nn.Conv2d(params.conv1_size, params.conv2_size, 4, stride=2, padding=1),
                                            nn.ReLU(True),
                                            nn.Conv2d(params.conv2_size, params.ego_num_chans, 3, stride=1, padding=1),
                                            nn.ReLU())            

        else:
            if params.skip_cnn_relu:
                self.conv_head = nn.Sequential(nn.Conv2d(num_inputs, params.conv1_size, 8, stride=4),
                                            nn.ReLU(True),
                                            nn.Conv2d(params.conv1_size, params.conv2_size, 4, stride=2),
                                            nn.ReLU(True),
                                            nn.Conv2d(params.conv2_size, params.ego_num_chans, 3, stride=1))
            else:
                self.conv_head = nn.Sequential(nn.Conv2d(num_inputs, params.conv1_size, 8, stride=4),
                                            nn.ReLU(True),
                                            nn.Conv2d(params.conv1_size, params.conv2_size, 4, stride=2),
                                            nn.ReLU(True),
                                            nn.Conv2d(params.conv2_size, params.ego_num_chans, 3, stride=1),
                                            nn.ReLU())  
        
        
        
        conv_input = torch.Tensor(torch.randn((1,) + input_shape))
        conv_out_size = self.conv_head(conv_input).nelement()
        
        self.conv_out_size = conv_out_size
        self.linear1 = nn.Linear(conv_out_size, params.hidden_size)
        # define global read
        global_read = nn.Sequential(nn.Conv2d(params.ego_num_chans, params.ego_num_chans, 3, stride=1, padding=1),
                                       nn.ReLU(True),
                                       nn.Conv2d(params.ego_num_chans, params.ego_num_chans, 4, stride=2, padding=0),
                                       nn.ReLU(True),
                                       nn.Conv2d(params.ego_num_chans, params.ego_num_chans, 4, stride=2, padding=0),
                                       nn.ReLU())

        global_read_input = torch.Tensor(torch.randn((1,) + (params.ego_num_chans, 
                                                         2*params.ego_half_size -1, 
                                                         2*params.ego_half_size -1,)))
        global_head_size = global_read(global_read_input).nelement()
    
         # This has to be defined as an extension as I do not know the size in advance
        global_extension = nn.Sequential(Lin_View(),
                          nn.Linear(global_head_size, 256),
                          nn.ReLU(True),
                          nn.Linear(256, params.ego_hidden_size),
                          nn.Tanh())      
        self.global_read = nn.Sequential(global_read, global_extension)
         
        # define query read
        self.context_linear = nn.Linear(params.hidden_size, params.ego_num_chans)
        
        w_size = params.ego_num_chans // 4
        cat1_size = params.ego_hidden_size + params.hidden_size + params.ego_num_chans
        cat2_size = cat1_size + w_size

        if params.nm_gru_op:
            self.wr = nn.Linear(cat2_size, w_size) # s, r, c, centre
            self.wh = nn.Linear(cat1_size, w_size) # s, r, c
            self.uh = nn.Linear(w_size, w_size) # centre hadamard
            self.wz = nn.Linear(cat2_size, w_size) # s, r, c, centre
        else:
            self.write_linear = nn.Linear(cat1_size + params.ego_num_chans, w_size)
          
            
        if params.recurrent_policy:
            assert params.use_lstm == False, 'Cannot have both GRU and LSTM!'
            self.gru = nn.GRUCell(params.hidden_size + params.ego_hidden_size, params.hidden_size)
   
        
        if self.params.nm_skip:
            self.critic_linear = nn.Linear(params.ego_hidden_size  + 2* params.ego_num_chans + params.hidden_size, 1)
            self.dist = Categorical(params.ego_hidden_size  + 2* params.ego_num_chans + params.hidden_size, params.num_actions)
        else:
            self.critic_linear = nn.Linear(params.ego_hidden_size  + 2* params.ego_num_chans, 1)
            self.dist = Categorical(params.ego_hidden_size  + 2* params.ego_num_chans, params.num_actions)
        
  
        self.train()
        self.reset_parameters()        
   
    def gru_op(self, state, global_read, context, sub_centres):
        cat1 = torch.cat([state, global_read, context], 1)
        cat2 = torch.cat([state, global_read, context, sub_centres], 1)
        
        r = F.sigmoid(self.wr(cat2))

        w_tilde = F.tanh(self.wh(cat1) + self.uh(r*sub_centres))
        z = F.sigmoid(self.wz(cat2))
        
        output = (1-z)* sub_centres + z*w_tilde
        
        return output
        
    def get_sub_vectors(self, vectors, orientations):
        inds = self.descretize_orientations(orientations)
        sub_map_size = self.params.ego_num_chans // 4
        outputs = []
        batch_size = vectors.size(0)
        
        for i in range(batch_size):
            vec = vectors[i, inds[i]:inds[i]+sub_map_size]
            outputs.append(vec.unsqueeze(0))
            
        return torch.cat(outputs,0)
        
    def query(self, query_vectors, ego_states):
        # tested
        batch_size = ego_states.size(0)
        scores = (ego_states* query_vectors.view(-1, self.params.ego_num_chans, 1, 1)).sum(1)
        scores_norm = F.softmax(1.0E-8+scores.view(batch_size,-1), dim=1).view(scores.size())
        weighted_states = ego_states*scores_norm.unsqueeze(1)
        
        contexts = weighted_states.view(batch_size, self.params.ego_num_chans, -1).sum(2)
        return contexts
    
    def descretize_orientations(self, orientations): 
        # TODO: this doesnt exactly give north south east west, it is off by 45 degrees
        return torch.floor((orientations)/ ( math.pi/2)).long()
    
    def local_write(self, ego_states, write_vectors, positions, origins, orientations):
        # Tested:
        # Correct mapping
        # Gradient backprop
 
        assert self.params.ego_num_chans % 4 == 0
        sub_map_size = self.params.ego_num_chans // 4
        batch_size = ego_states.size(0)
        map_half_size = ego_states.size(-1) // 2
        
        deltas = positions - origins
        discrete_deltas = (self.discretize(deltas) + map_half_size).long()
        NSEW_orientations = self.descretize_orientations(orientations)
        NSEW_indices = NSEW_orientations * sub_map_size
        output_vector_maps = []
        
        for i in range(batch_size):
            output_map = torch.zeros_like(ego_states[i])
            ind_i, ind_j = discrete_deltas[i]
            output_map[NSEW_indices[i]:NSEW_indices[i]+sub_map_size, ind_i, ind_j] = write_vectors[i]
            output_vector_maps.append(output_map)
            
        output_vector_maps = torch.stack(output_vector_maps)
        output_states = ego_states + output_vector_maps
        
        return output_states
    
    def local_read(self, ego_states, deltas):
        batch_size = ego_states.size(0)
        map_half_size = ego_states.size(-1) // 2
        
        discrete_deltas = (self.discretize(deltas) + map_half_size).long()
        
        output_vectors = []
        
        for i in range(batch_size):
            ind_i, ind_j = discrete_deltas[i]
            read_vector = ego_states[i, :, ind_i, ind_j]
            output_vectors.append(read_vector)
        
        return torch.stack(output_vectors)
        
  
    def discretize(self, values):
        return torch.round(values / self.params.ego_bin_dim).long()
    
    def shift(self, tensor, shift_i, shift_j):
        c, h, w = tensor.size()
        out_tensor = tensor
        if shift_i ==0 and shift_j == 0:
            return out_tensor
       
        # i and j are independant so can be handled separately      
        if shift_i > 0:
            sub = out_tensor[:,shift_i:,:]
            pad = torch.zeros(c, shift_i, h).to(self.device)
            out_tensor = torch.cat([sub, pad], 1)
        elif shift_i < 0:
            sub = out_tensor[:,:shift_i,:]
            pad = torch.zeros(c, shift_i.abs(), h).to(self.device)
            out_tensor = torch.cat([pad, sub], 1)
        
        if shift_j > 0:
            sub = out_tensor[:,:,shift_j:]
            pad = torch.zeros(c, w, shift_j).to(self.device)
            out_tensor = torch.cat([sub, pad], 2)
            
        elif shift_j < 0:
            sub = out_tensor[:,:,:shift_j]
            pad = torch.zeros(c, w, shift_j.abs()).to(self.device)
            out_tensor = torch.cat([pad, sub], 2)
                   
        return out_tensor

    def counter_transform(self, ego_states, deltas):
        # test: grad & function
        discrete_deltas = self.discretize(deltas)
        state_size = ego_states.size(0)
        # padding = ego_states.size(-1) // 2
        # groups = ego_states.size(1) # num chans in ego map
        
        out_states = []
        b, c, h, w = ego_states.size()
        # TODO make this a single operation
        for i in range(state_size):
            ego_state = ego_states[i]
            
            ind_i, ind_j = discrete_deltas[i]
            
            out_state = self.shift(ego_state, ind_i, ind_j)                
            out_states.append(out_state.unsqueeze(0))
        
        return torch.cat(out_states, 0)

    # TODO: update this for use with kwargs
    def forward(self, inputs, states, masks, 
                pred_depth=False, 
                ego_states=None, ego_depths=None, 
                pos_deltas_origins=None, states2=None,
                prev_obs=None, prev_actions=None, curiousity=None):
        
        assert masks is not None
        assert ego_states is not None
        assert pos_deltas_origins is not None
        
        #print(masks)
        half_size = ego_states.size(-1) // 2
        conv_out = self.conv_head(inputs * (1.0/255.0))
        conv_out = conv_out.view(-1, self.conv_out_size)
        x = self.linear1(conv_out)
        x = F.relu(x)
        
        if inputs.size(0) == ego_states.size(0): # 1 observation
            positions, orientations = pos_deltas_origins[:,:2],  pos_deltas_origins[:,2]
            origins = pos_deltas_origins[:,6:8]
                
            global_deltas = positions - origins
            # I translate to ego viewpoint for reading
            translated_ego_states = self.counter_transform(ego_states* masks.view(-1,1,1,1), global_deltas)

            ego_global_read = self.global_read(translated_ego_states)#.view(inputs.size(0), -1)
            
            
            
            states = self.gru(torch.cat([x, ego_global_read], 1), states * masks)
            
            query_vectors = self.context_linear(states)
            
            context_read = self.query(query_vectors, translated_ego_states)

            central_vector = translated_ego_states[:, :, half_size, half_size]
            
            if self.params.nm_gru_op:
                sub_central_vectors = self.get_sub_vectors(central_vector, orientations)
                write_vectors = self.gru_op(x, ego_global_read, context_read, sub_central_vectors)
            else:
                write_vectors = self.write_linear(torch.cat([x, ego_global_read, context_read, central_vector], 1))
            # normalization to stop exploding values    
            write_vectors = write_vectors / write_vectors.norm(dim=1, keepdim=True)
            
            ego_states = self.local_write(ego_states* masks.view(-1,1,1,1), write_vectors,
                                          positions, origins, orientations)
            
            local_read = self.local_read(ego_states, global_deltas)            
            x = torch.cat([context_read, ego_global_read, local_read], 1)
            
            if self.params.nm_skip:
                x = torch.cat([context_read, ego_global_read, local_read, states], 1)     
            else:    
                x = torch.cat([context_read, ego_global_read, local_read], 1) 
            
            
        else:
            
            x = x.view(-1, ego_states.size(0), *x.size()[1:])
            masks = masks.view(-1, ego_states.size(0), 1)
            
            outputs = []
            out_states = []
            for i in range(x.size(0)):
                positions, orientations = pos_deltas_origins[i, :, :2],  pos_deltas_origins[i, :, 2]
                origins = pos_deltas_origins[i, :,6:8]
   
                global_deltas = positions - origins
                # I translate to ego viewpoint for reading
                translated_ego_states = self.counter_transform(
                        ego_states * masks[i].unsqueeze(-1).unsqueeze(-1), global_deltas)

                ego_global_read = self.global_read(translated_ego_states) 
                
                states = self.gru(torch.cat([x[i], ego_global_read], 1), states * masks[i])
                
                query_vectors = self.context_linear(states)
                
                context_read = self.query(query_vectors, translated_ego_states)
                
                central_vector = translated_ego_states[:, :, half_size, half_size]
                
                if self.params.nm_gru_op:
                    sub_central_vectors = self.get_sub_vectors(central_vector, orientations)
                    write_vectors = self.gru_op(x[i], ego_global_read, context_read, sub_central_vectors)
                else:
                    write_vectors = self.write_linear(torch.cat([x[i], ego_global_read, context_read, central_vector], 1))
                # normalization to stop exploding values
                write_vectors = write_vectors / write_vectors.norm(dim=1, keepdim=True)
                
                ego_states = self.local_write(ego_states* masks[i].unsqueeze(-1).unsqueeze(-1), write_vectors,
                                              positions, origins, orientations)

                local_read = self.local_read(ego_states, global_deltas)
                if self.params.nm_skip:
                    out = torch.cat([context_read, ego_global_read, local_read, states], 1)     
                else:    
                    out = torch.cat([context_read, ego_global_read, local_read], 1) 

                outputs.append(out)
                out_states.append(states)
                
            x = torch.cat(outputs, 0)    
            states = torch.cat(out_states, 0)
            
        result = {'values': self.critic_linear(x), 
                  'x': x, 
                  'states': states, 
                  'ego_states': ego_states}        
        
        return result
        
        
    @property
    def state_size(self):
        if hasattr(self, 'gru') or hasattr(self, 'lstm'):
            return self.params.hidden_size
        else:
            return 1

    def reset_parameters(self):
        self.apply(weights_init)

        relu_gain = nn.init.calculate_gain('relu')
        tanh_gain = nn.init.calculate_gain('tanh')
        sigmoid_gain = nn.init.calculate_gain('sigmoid') # = 1
        for i in range(0, 6, 2):
            self.conv_head[i].weight.data.mul_(relu_gain)
            self.global_read[0][i].weight.data.mul_(relu_gain)
            
        for i in range(1, 3, 2):
            self.global_read[1][i].weight.data.mul_(relu_gain)                                     
            
        self.global_read[1][3].weight.data.mul_(tanh_gain)
            
        self.linear1.weight.data.mul_(relu_gain)
        
        if self.params.nm_gru_op:
            self.wr.weight.data.mul_(sigmoid_gain)
            self.wh.weight.data.mul_(tanh_gain)
            self.uh.weight.data.mul_(tanh_gain)
            self.wz.weight.data.mul_(sigmoid_gain)   
            
            self.wr.bias.data.fill_(0)
            self.wh.bias.data.fill_(0)
            self.uh.bias.data.fill_(0)
            self.wz.bias.data.fill_(0)    
            
        if hasattr(self, 'gru'):
            orthogonal(self.gru.weight_ih.data)
            orthogonal(self.gru.weight_hh.data)
            self.gru.bias_ih.data.fill_(0)
            self.gru.bias_hh.data.fill_(0)
            if self.params.learn_init_state:
                self.init_state = nn.Parameter(torch.randn(1, self.params.hidden_size) * 0.05)  
                
        if hasattr(self, 'lstm'):
            orthogonal(self.lstm.weight_ih.data)
            orthogonal(self.lstm.weight_hh.data)
            self.lstm.bias_ih.data.fill_(0)
            self.lstm.bias_hh.data.fill_(0)
            if self.params.gate_init:
                self.lstm.bias_ih.data[self.hidden_size: 2*self.params.hidden_size].fill_(1)
                print('hidden gate initialized')            
            
            
                
        if self.dist.__class__.__name__ == "DiagGaussian":
            self.dist.fc_mean.weight.data.mul_(0.01)     

