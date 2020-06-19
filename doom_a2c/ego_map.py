#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu April 19 10:00:00 2018

@author: anonymous

 The neural map Class defines a differentiable Egocentric map of the world


"""
import random
import torch
from torch import Tensor
import torch.nn.functional as F
import math
def one_plus(x):
    return 1 + (1+x.exp()).log()

class EgoMap():
    """
        The egocentric mapping class.
        The map itself is a Tensor, this class may contain many maps for multiple 
        agents
    """
    def __init__(self, params):
        self.params = params
        self.bin_dim = params.ego_bin_dim
        self.half_size = params.ego_half_size
        self.discrete_size = int(params.ego_half_size*2)
        
        self.half_world_size = self.bin_dim * self.half_size
        self.height_reject = params.ego_height_reject
        self.use_gpu = params.cuda
        self.alpha = 0.9
        self.origin_stack = [] # a stack to backup and restore origins
        self.num_channels = params.ego_num_chans
        if params.ego_curiousity:
            self.num_channels += 1
        self.num_simulations = params.num_environments
        
        self.projection_matrix = params.ego_proj_mat
    
        # A matrix holding the screen x,y coords for the conv outputs
        subtract = 0 # if padding is not used this is reduced
        if not params.new_padding:
            subtract = 4

        self.conv_feature_width = params.screen_width // 8 - subtract
        self.conv_feature_height = params.screen_height // 8 - subtract


        self.uv_matrix = self.create_uv_matrix(self.conv_feature_height, 
                                               self.conv_feature_width)



        top_corner_mask = torch.ones(1, 1, self.conv_feature_height, self.conv_feature_width)
        top_corner_mask[0,0,0,0] = 0
        self.top_corner_mask = top_corner_mask
        
        # as the screen is not the same size as the ego map that we project to, 
        # we must calculate the bin locations of the screen and the upsample the screen
        self.screen_bin_width = self.bin_dim * 2.0 * self.half_size / self.conv_feature_width
        self.screen_bin_height = self.bin_dim * self.half_size / self.conv_feature_height
    
        # pre-computed variables to speed up ego mapping
        self.num_inds_screen = self.conv_feature_height*self.conv_feature_width
        self.num_inds_ego = self.half_size*self.discrete_size
        self.uv_index_matrix = torch.LongTensor(list(range(self.num_inds_screen)))
        self.shift_thresh = params.shift_thresh
    
        self.index_shift = (torch.zeros(self.conv_feature_height, self.conv_feature_width) + 
                       torch.Tensor(list(range(0, self.conv_feature_height))).unsqueeze(1) * 
                       self.num_inds_ego).view(1,-1).long() # this ensure mapping to separate indices
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if params.query_position:
            self.create_xy_planes()
        
        self.index_map = torch.zeros(self.num_inds_ego * self.conv_feature_height).long()

        self.aug_matrices = torch.zeros(params.num_environments, 2,3)

        self.set_device(self.device)
        
    def set_device(self, device):
        self.uv_matrix = self.uv_matrix.to(device)
        self.index_shift = self.index_shift.to(device)
        self.top_corner_mask = self.top_corner_mask.to(device)
        self.uv_index_matrix = self.uv_index_matrix.to(device)
        self.index_map = self.index_map.to(device)
        self.aug_matrices = self.aug_matrices.to(device)

        if hasattr(self, 'xy_planes'):
            self.xy_planes = self.xy_planes.to(device)
        
          
    def create_uv_matrix(self, height, width):
        def get_screen_coords(v, u, height, width):
            # normalized so that a box at centre is in (-1,1) for both height and width
            # shifted by 1/width to get bin centres rather than corners
            
            width_scalar = 1.3885
            height_scalar = 1.0
            
            if not self.params.new_padding:
                width_scalar *= (4.0/7.0) # no padding will reduce edges by 2 on each size
                height_scalar *= (4.0/8.0) # no padding will reduce edges by 2 on each size
            
            xx = (2.0*u / width - 1.0 + 1.0/width)  * width_scalar # aspect ratio normalization
            yy = (2.0*v / height - 1.0 + 1.0/height) * height_scalar
            return [yy, xx]        
        uv_matrix = torch.zeros(height, width, 3)
        
        for v in range(height):
            for u in range(width):
                uv_matrix[v, u,:] = torch.Tensor(get_screen_coords(v, u, height, width)+[1.0])
                
        #uv_matrix[:,:,0] /= 1.0 # normalization required for width projection
        return torch.Tensor(uv_matrix).view(1, -1, 3)


    def create_memory(self, batch_size, test=False):
        return torch.zeros(batch_size, self.num_channels, self.discrete_size, self.discrete_size)

    def create_xy_planes(self):
        ego_size = self.params.ego_half_size * 2
        x_plane = torch.linspace(-1,1, ego_size).view(1, 1, ego_size).repeat(1, ego_size, 1)
        y_plane = x_plane.transpose(2,1)
        self.xy_planes = torch.cat([x_plane, y_plane], 0).view(1, 2, ego_size, ego_size)

    def project(self, depths, batch_size):        
        depths_repeated = depths.contiguous().view(batch_size, -1, 1).repeat(1,1,3)
        yxz_coords = depths_repeated * self.uv_matrix

        return yxz_coords     

    def rotate_translate(self, ego_memory, dxs, dys, dthetas):
        norm_dxs = dxs / (self.half_world_size)
        norm_dys = dys / (self.half_world_size)
        
        norm_dthetas = dthetas  # change degrees to radians 
        #aug_matrices = torch.zeros(dxs.size(0), 2,3).to(self.device)
        #self.aug_matrices.zero_()

        self.aug_matrices[:,0,0] = torch.cos(norm_dthetas)
        self.aug_matrices[:,0,1] = torch.sin(norm_dthetas)
        self.aug_matrices[:,1,0] = -torch.sin(norm_dthetas)
        self.aug_matrices[:,1,1] = torch.cos(norm_dthetas)
        self.aug_matrices[:,0,2] = -norm_dxs # minus because we translate in opposite direction
        self.aug_matrices[:,1,2] = -norm_dys        

        flow_field = F.affine_grid(self.aug_matrices, ego_memory.size())
        
        return F.grid_sample(ego_memory, flow_field)
    

    def get_ego_tensor_cw(self, x, depths):

        batch_size, vector_size, height, width, = x.size()
        yxzs = self.project(depths, batch_size)

        X = yxzs[:, :, 1] / self.bin_dim + self.half_size
        Z = self.half_size - (yxzs[:, :, 2] / self.bin_dim) # Flip indices at z axis is in opposite direction
        Y = yxzs[:, :, 0]
        

        x = x * self.top_corner_mask # zero the top left corner
        
        Xj = torch.round(X).long() # Find nearest index
        Zi = torch.round(Z).long() # Find nearest index
        # the top left corner will be used to map the 
        # outliers include height masking
        #outliers = (Xj < 0) + (Xj >= self.discrete_size) + (Zi < 0) + (Zi >= self.half_size) + (Y < -65) + (Y >  45)
        if self.height_reject:
            outliers = (Xj < 0) + (Xj >= self.discrete_size) + (Zi < 0) + (Zi >= self.half_size) + (Y < -65) + (Y >  40) # reject the floor and the ceiling
        else:
            outliers = (Xj < 0) + (Xj >= self.discrete_size) + (Zi < 0) + (Zi >= self.half_size)
                    
        Xj[outliers > 0] = 0
        Zi[outliers > 0] = 0        
        
        ego_maps = []
        weights = []
        xz_indices = Zi*self.discrete_size + Xj
        x = x.view(-1, 1, vector_size, self.conv_feature_height * self.conv_feature_width).contiguous().repeat(1, self.conv_feature_height, 1, 1)

        for b in range(batch_size): 
            #index_map = torch.zeros(self.num_inds_ego * self.conv_feature_height).long().to(self.device)
            self.index_map.zero_()
            self.index_map.put_(xz_indices[b] + self.index_shift,  self.uv_index_matrix)
              
            ego_features = torch.gather(x[b], 2, 
                                self.index_map.view(self.conv_feature_height, 1, self.half_size * self.discrete_size).repeat(1, vector_size, 1)) 
            
            weight = 1.0 - ego_features.sum(1).eq(0.0)
            
            ego_maps.append(ego_features.sum(0).view(1, self.num_channels, self.half_size, self.discrete_size))
            
            weights.append(weight.sum(0).view(1, 1, self.half_size, self.discrete_size))
           
        return torch.cat(ego_maps, dim=0), torch.cat(weights, dim=0).float().detach()
    
        
    def ego_mapper(self, x, depths, ego_memory, xxs, yys, thetas, masks, origin_x, origin_y, debug=False):
        world_dxs = xxs - origin_x
        world_dys = yys - origin_y
        dthetas = thetas #- self.start_thetas

        h = torch.sqrt(world_dxs**2 + world_dys**2)
        world_thetas = torch.atan2(world_dys, world_dxs)
        theta_hats = thetas - world_thetas
       
        dxs = h*torch.sin(theta_hats)
        dys = h*torch.cos(theta_hats) 
       
        ego_observations, ego_weights = self.get_ego_tensor_cw(x, depths)
        ego_weights[ego_weights == 0] = 1
        ego_observations = ego_observations / ego_weights
        
        padded_ego_obs = torch.cat([ego_observations, torch.zeros_like(ego_observations)], dim=2)
        
         # pi/2 because the angle is measured from the x-axis but observations are initially projected upwards
        transformed_observations = self.rotate_translate(padded_ego_obs, dxs, -dys, -dthetas + math.pi/2)
        # find locations in the tranformed map that are non-zero
        #obs_mask = transformed_observations.sum(dim=1).nonzero().float() 
        
        # identify the locations in the ego memory which contain non-zero values
        memory_mask = ego_memory.detach().sum(dim=1)
        memory_mask[1 - (memory_mask == 0)] = (1.0-self.alpha)
        memory_mask[memory_mask == 0] = 1.0
        
        # identify the locations in the new observation that contain non-zero values       
        obs_mask = transformed_observations.detach().sum(dim=1)
        obs_mask[1 - (obs_mask == 0)] = self.alpha
        obs_mask[obs_mask == 0] = 1.0

        return (ego_memory*obs_mask.unsqueeze(1) + 
                transformed_observations*memory_mask.unsqueeze(1))
        
    def rotate_for_read(self, ego_memory, xxs, yys, thetas, origin_x, origin_y):
                
        world_dxs = xxs - origin_x
        world_dys = yys - origin_y
        dthetas = thetas #- self.start_thetas
        
        transformed_maps = self.rotate_translate(ego_memory, -world_dxs, 
                                            world_dys, dthetas)        
              
        return transformed_maps
            
 
    def query(self, query_vectors, ego_states):
        # gradient checked
        if self.params.ego_query_scalar:
            query_vectors = query_vectors[:, :-1]
            scalar = query_vectors[:, -1:]
            scalar = one_plus(scalar)
        else:
            scalar = 1.0
        batch_size = ego_states.size(0)
        #print(ego_states.norm())
        query_vectors = query_vectors.view(-1, self.num_channels, 1, 1)
        if self.params.ego_query_cosine:            
            query_norm = query_vectors / (query_vectors.norm(2, dim=1, keepdim=True) + 1E-8)
            states_norm = ego_states / (ego_states.norm(2, dim=1, keepdim=True) + 1E-8)            
            scores = (states_norm* query_norm).sum(1)            
        else:
            scores = (ego_states* query_vectors).sum(1)
            
        scores_norm = F.softmax(scores.view(batch_size,-1)*scalar,1).view(scores.size())
        if self.params.viz_att:
            self.attention = scores_norm
        #print('scores')
        #print(scores_norm.sum())
        weighted_states = ego_states*scores_norm.unsqueeze(1)
        
        contexts = weighted_states.view(batch_size, self.num_channels, -1).sum(2)
        
        if self.params.viz_att:
            self.beta = scalar            
        
        if self.params.query_position:

            weighted_positions = (self.xy_planes*scores_norm.unsqueeze(1)).view(batch_size,2, -1).sum(2)
            contexts = torch.cat([contexts, weighted_positions], 1)
            
            if self.params.viz_att:
                self.weighted_positions = weighted_positions
            
            
        #print(contexts)
        return contexts
        
def test_ego_query():
    params = parse_game_args()
    params.ego_num_chans = 4
    params.ego_half_size = 4
    params.query_position = False
    params.ego_query_cosine = False
    ego_map = EgoMap(params)
    
    
    # Checking of gradient
    # Ego mapper
    
    # rotate for read
    # query
    states = torch.zeros(1,params.ego_num_chans,params.ego_half_size*2,params.ego_half_size*2)
    
    vectors = torch.randn(1,params.ego_num_chans)
    states[0,:,0,0] = vectors[0]
    states.requires_grad= True
    tt = states[0,:, 6,7]
    context = ego_map.query(vectors, states)
    

    
    context
    vectors
    attention
    
    context_grad = torch.zeros_like(context)
    context_grad[0,3] = 1.0
    
    torch.autograd.backward(context, context_grad)
    
    print(states.grad[0,:,0,0])
                
if __name__ == '__main__':

    from doom_a2c.arguments import parse_game_args 
    import matplotlib.pyplot as plt
    params = parse_game_args()
    params.ego_num_chans = 4
    params.ego_half_size = 4
    params.query_position = False
    params.ego_query_cosine = False
    ego_map = EgoMap(params)
    
    
    # Checking of gradient
    # Ego mapper
    
    # rotate for read
    
    states = torch.zeros(1,params.ego_num_chans,params.ego_half_size*2,params.ego_half_size*2)
    

     
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # aa = ego_map.uv_matrix.numpy()
    # from environments import DoomEnvironment
    # import cv2
    # from cv2 import resize
    # params.scenario_dir = '../resources/scenarios/'
    # env = DoomEnvironment(params, get_extra_info=True)
    
    # ego_memory = ego_map.create_memory(16)
    
    # ## Testing for Christian EgoMap method
    # env.step(4)
    # obs = env.get_observation()
    # depth = env.state.depth_buffer
    # resized_obs =   resize(obs.transpose(1,2,0), 
    #                 (params.screen_width // 8, params.screen_height // 8), cv2.INTER_AREA)     
    # resized_depth =   resize(depth, (params.screen_width // 8, 
    #                                  params.screen_height // 8), cv2.INTER_AREA)  
    # resized_depth_scaled = resized_depth * 7.4 + 7.0    
    # example_input = torch.rand(16,32,8,14)
    
    
    # depths = Tensor(resized_depth_scaled).unsqueeze(0).repeat(16,1,1)

    # x, depths, ego_memory, dxs, dys, dthetas = example_input, depths, ego_memory, Tensor([0.5]*16), Tensor([0.5]*16), Tensor([0.5]*16)

    # #ego_memory = ego_map.ego_mapper(x, depths, ego_memory, dxs, dys, dthetas)
    

