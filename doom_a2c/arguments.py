#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 10:37:33 2018

@author: anonymous


PongNoFrameskip-v4

"""

import argparse
def parse_game_args():
    """ Defines the arguments used for both training and testing the network"""
    
    parser = argparse.ArgumentParser(description='Parameters')

    # =========================================================================
    #               Environment Parameters
    # =========================================================================
    parser.add_argument('--simulator', type=str, default="doom", help='The environment')
    parser.add_argument('--scenario', type=str, default='health_gathering_supreme.cfg', help='The scenario')
    parser.add_argument('--test_scenario', type=str, default='', help='The scenario used for testing')
    parser.add_argument('--screen_size', type=str, default='320X180', help='Size of Screen, width x height')    
    parser.add_argument('--screen_height', type=int, default=64, help='Height of the screen')
    parser.add_argument('--screen_width', type=int, default=112, help='Width of the screen')
    parser.add_argument('--num_environments', type=int, default=16, help='the number of parallel enviroments')
    parser.add_argument('--limit_actions', default=False, action='store_true', help='limited the size of the action space to F, L, R, F+L, F+R')
    parser.add_argument('--use_depth', action='store_true', default=False,  help='Use the Depth Buffer') 
    parser.add_argument('--scenario_dir', type=str, default='resources/scenarios/', help='location of game scenarios')    
    parser.add_argument('--test_scenario_dir', type=str, default='', help='location of game scenarios')    
    parser.add_argument('--show_window', type=bool, default=False, help='Show the game window')
    #parser.add_argument('--decimate', type=bool, default=True, help='Subsample the observations')
    parser.add_argument('--resize', type=bool, default=True, help='Use resize for decimation rather ran downsample')
    parser.add_argument('--norm_obs', dest='norm_obs',default=True, action='store_false', help='Divide the obs by 255.0')
    parser.add_argument('--multimaze', default=False, action='store_true', help='are there multiple maze environments')
    parser.add_argument('--num_mazes_train', type=int, default=16, help='the number of training mazes, only valid for multimaze')
    parser.add_argument('--num_mazes_test', type=int, default=1, help='the number of testing mazes, only valid for multimaze')
    parser.add_argument('--disable_head_bob', action='store_true', default=False, help='disable head bobbing')    
    parser.add_argument('--use_shaping', action='store_true', default=False, help='include shaping rewards')    
    parser.add_argument('--fixed_origin', action='store_true', default=False, help='fix origin to 0,0')    
    parser.add_argument('--fixed_scenario', action='store_true', default=False, help='whether to stay on a fixed scenario')    
    parser.add_argument('--use_pipes', action='store_true', default=False, help='use pipes instead of queue for the environment')    
    parser.add_argument('--sleep', type=float, default=0.0, help='sleep the simulator during startup')    
    parser.add_argument('--num_actions', type=int, default=0, help='size of action space')
    
    # =========================================================================
    #               Model Parameters
    # =========================================================================
    parser.add_argument('--hidden_size', type=int, default=512, help='LSTM / GRU hidden size')
    parser.add_argument('--conv_filters', type=int, default=32, help='Number of convolutional filters' )   
    parser.add_argument('--predict_depth', default=False, action='store_true', help='make depth predictions')
    parser.add_argument('--reload_model', type=str, default='', help='directory and iter of model to load dir,iter')
    parser.add_argument('--model_checkpoint', type=str, default='', help='the name of a specific model to evaluate, used when making videos')
    parser.add_argument('--recurrent_policy', action='store_true', default=False, help='use a gru recurrent policy')    
    parser.add_argument('--use_lstm', action='store_true', default=False, help='use a lstm recurrent policy')   
    parser.add_argument('--gate_init', action='store_true', default=False, help='Initialize the LSTM forget to zero, gru reset to -1')    
    parser.add_argument('--learn_init_state', action='store_true', default=False, help='Learning the initial state of the gru / lstm')    
    parser.add_argument('--gru_forget_init', action='store_true', default=False, help='proper init for gru cell')    
    parser.add_argument('--gru_bias_range', type=float, default=0.5, help='range for random init of gru cell biases')    
    parser.add_argument('--gru_skip', action='store_true', default=False, help='include gru skip')

    parser.add_argument('--new_padding', action='store_true', default=False, help='pad the edges appropriately')
    parser.add_argument('--centre_values', action='store_true', default=False, help='normalize to +/- 1.0')
    parser.add_argument('--skip_fc', action='store_true', default=False, help='skip this fc layer')

    parser.add_argument('--conv1_size', type=int, default=32, help='Number of filters in conv layer 1')
    parser.add_argument('--conv2_size', type=int, default=64, help='Number of filters in conv layer 2')
    parser.add_argument('--conv3_size', type=int, default=32, help='Number of filters in conv layer 3')
    parser.add_argument('--action_prediction',  action='store_true', default=False, help='include an action prediction cnn layer and loss')
    parser.add_argument('--action_lambda',  type=float, default=0.1, help='penalty on action prediction') 
    parser.add_argument('--action_kl', action='store_true', default=False, help='use KL divergence as loss for action prediction')
    parser.add_argument('--loop_detect', action='store_true', default=False, help='predict whether the agent has visitic a location before')
    parser.add_argument('--loop_detect_lambda', type=float, default=0.1, help='loop weighting') 
    parser.add_argument('--image_scalar', type=float, default=255.0, help='image scaling factor') 
    
    
    parser.add_argument('--conv_head_checkpoint', type=str, default='', help='name of save conv head to load')
    parser.add_argument('--linear_head_checkpoint',  action='store_true', default=False, help='whether to also load the linear layer( must be combiend with cnn load')
    
    parser.add_argument('--stacked_gru', action='store_true', default=False, help='use a two layered gru')
    
    parser.add_argument('--pretrained_vae', type=str, default='',  help='use a pretraind feature extractor')
    
    parser.add_argument('--depth_as_obs', action='store_true', default=False, help='include depth buffer in input')
    parser.add_argument('--pos_as_obs', action='store_true', default=False, help='include postion in input')
    
    
    
    # =========================================================================
    #               Training Parameters 
    # =========================================================================    
    parser.add_argument('--learning_rate', type=float, default=7e-4, help='training learning rate')
    parser.add_argument('--momentum', type=float, default=0.0, help='optimizer momentum')
    
    parser.add_argument('--gamma', type=float, default=0.99, help='reward discount factor')
    parser.add_argument('--frame_skip', type=int, default=4, help='number of frames to repeat last action')
    parser.add_argument('--train_freq', type=int, default=4, help='how often the model is updated')
    parser.add_argument('--train_report_freq', type=int, default=100, help='how often to report the train loss')
    parser.add_argument('--max_iters', type=int, default=5000000, help='maximum number of training iterations')
    parser.add_argument('--eval_freq', type=int, default=5000, help='how often the model is evaluated, in games')
    parser.add_argument('--eval_games', type=int, default=10, help='how often the model is evaluated, in games')
    parser.add_argument('--cuda', type=bool, default=False, help='Use the GPU?')
    parser.add_argument('--model_save_rate', type=int, default=50000, help='How often to save the model in iters')
    parser.add_argument('--pretrained_head',type=str, default='', help='Name of pretrained convolutional head')
    parser.add_argument('--freeze_pretrained', type=bool, default=True, help='Whether to freeze the weights in pretrained head')
    parser.add_argument('--eps', type=float, default=1e-5, help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument('--alpha', type=float, default=0.99, help='RMSprop optimizer alpha (default: 0.99)')    
    parser.add_argument('--use-gae', action='store_true', default=False, help='use generalized advantage estimation')
    parser.add_argument('--tau', type=float, default=0.95, help='gae parameter (default: 0.95)')
    parser.add_argument('--entropy_coef', type=float, default=0.01, help='entropy term coefficient (default: 0.01)')
    parser.add_argument('--value_loss_coef', type=float, default=0.5, help='value loss coefficient (default: 0.5)')
    parser.add_argument('--max_grad_norm', type=float, default=0.5, help='max norm of gradients (default: 0.5)')    
    parser.add_argument('--num_steps', type=int, default=5, help='number of forward steps in A2C (default: 5)')
    parser.add_argument('--num_stack', type=int, default=1,help='number of frames to stack (default: 1)')
 
    parser.add_argument('--num_frames', type=int, default=10000000, help='total number of frames')          
    parser.add_argument('--depth_coef', type=float, default=0.01, help='weighting for depth loss')
    parser.add_argument('--no_reward_average', default=True, action='store_true', help='switch of reward averaging during frame skip')
    parser.add_argument('--use_em_loss', default=False, action='store_true', help='Use the discrete EM loss, optimal transport for depth preds')
    parser.add_argument('--skip_eval', default=False, action='store_true',help='skip the evaluation process')
    parser.add_argument('--stoc_eval', default=False, action='store_true',help='evaluate stochastically')
    
    
    # =========================================================================
    #               Test Parameters 
    # =========================================================================        
    parser.add_argument('--model_dir', type=str, default='', help='Location of model directory for test/train evaluation')
    parser.add_argument('--eval_dir_models', type=str, default='', help='dir and models eg /home/.../models/,modelA,modelB,modelC')
    
        
    # =========================================================================
    #               Logging Parameters 
    # =========================================================================
    parser.add_argument('--user_dir', type=str, default='user', help='Users home dir name')
    parser.add_argument('--log_interval', type=int, default=100, help='How often to log')
    parser.add_argument('--job_id', type=int,   default=12345,      help='the slurm job queue id')
    parser.add_argument('--test_name', type=str,   default='test_000.sh',      help='name of the test')
    parser.add_argument('--use_visdom', default=False, action='store_true', help='use visdom for live visualization')
    parser.add_argument('--visdom_port', type=int,   default=8097,      help='the port number visdom will use')
    parser.add_argument('--visdom_ip', type=str,   default='http://10.0.0.1',      help='the ip address of the visdom server') 

    # =============================================================================
    #     EgoMap Parameters
    # =============================================================================
    # chans, map_width, map_dim, ...

    parser.add_argument('--ego_bin_dim', type=float, default=64.0, help='The size in world units of the bins of the egomap')
    parser.add_argument('--ego_half_size', type=int, default=24, help='The half width/height of the egomap tensor')
    parser.add_argument('--ego_num_chans', type=int, default=32, help='The number of channels in the egomap')
    parser.add_argument('--ego_proj_mat', type=str, default='42', help='The parameters of the projection matrix')
    parser.add_argument('--ego_model', type=str, default='', help='The type of ego network to use, leave empty to run the baseline agent')
    parser.add_argument('--ego_height_reject', default=False, action='store_true', help='Reject the ground and ceiling from egomap projection')
    parser.add_argument('--ego_skip', default=False, action='store_true', help='include the skip connection')
    parser.add_argument('--ego_query', default=False, action='store_true', help='use a query vector')
    parser.add_argument('--reduce_blur', default=False, action='store_true', help='store a world map to reduce blurring')
    parser.add_argument('--shift_thresh', default=128.0, type=float, help='threshold to move the map')
    parser.add_argument('--ego_hidden_size', default=32, type=int, help='size of the ego read linear layer')
    parser.add_argument('--skip_world_shift', default=False, action='store_true', help='do not move the global map (for debugging)')
    parser.add_argument('--merge_later', default=False, action='store_true', help='concatenate the ego output later')
    parser.add_argument('--ego_use_tanh', default=False, action='store_true', help='use a tanh non-linearity for the egomap output')
    parser.add_argument('--query_relu', default=False, action='store_true', help='use a relu on the query vector')
    parser.add_argument('--skip_cnn_relu', default=False, action='store_true', help='do not use a relu activation on mapped egomap vectors')
    parser.add_argument('--ego_skip_global', default=False, action='store_true', help='skip the global read of the egomap')
    parser.add_argument('--query_position', default=False, action='store_true', help='use the attention distribution to query position')
    parser.add_argument('--ego_query_cosine', default=False, action='store_true', help='use cosine distance during ego query')
    parser.add_argument('--ego_query_scalar', default=False, action='store_true', help='use a learnable scalar to change the query probs')
    parser.add_argument('--viz_att', default=False, action='store_true', help='for viewing the egomap attention')
    parser.add_argument('--ego_curiousity', default=False, action='store_true', help='forward modelling for curiousity based prediction')
    parser.add_argument('--forward_model_name', default='', type=str, help='name of the forward model checkpoint')
    
   
    # =============================================================================
    #     Neural Map Parameters
    # =============================================================================    
    parser.add_argument('--neural_map', type=str, default='', help='Use the Neural Map model architecture, leave empty to run the baseline agent')
    parser.add_argument('--nm_gru_op',  default=False, action='store_true', help='GRU update as detailed in paper')
    parser.add_argument('--nm_skip',  default=False, action='store_true', help='include a skip connection of hidden state')
    
    
    
    params = parser.parse_args()
    params.test_name = params.test_name[:-3]



    print(params)
    
    return params
       
    
    
if __name__ == '__main__':
    params = parse_game_args()
    print(params)
    print(params.action_size)
    import os
    print(os.listdir(params.scenario_dir))
    print(params.scenario)
 
