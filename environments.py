#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 11:03:06 2018

@author: anonymous
"""
from math import isclose
import itertools
import random
from vizdoom import DoomGame, ScreenResolution, GameVariable, Button, AutomapMode, Mode, doom_fixed_to_double
import numpy as np
from cv2 import resize
import cv2
import math

class DoomEnvironment():
    """
        A wrapper class for the Doom Maze Environment
    """
    class PlayerInfo():
        """
            Small class to hold player position info etc
        
        """
        def __init__(self, x, y, theta, fixed_origin=False):
            self.x = x
            self.y = y
            self.theta = theta # in radians
            self.starting_theta = theta
                
            self.dx, self.dy, self.dtheta = 0.0, 0.0, 0.0
            self.vx, self.vy, self.dv = 0.0, 0.0, 0.0
            
            if fixed_origin:
                self.origin_x = 0.0
                self.origin_y = 0.0
            else:
                self.origin_x = x
                self.origin_y = y
                
            self.history = [(x,y)]
            
        def update(self, x, y, theta):         
            self.history.append((x,y))
            self.dtheta = theta - self.theta
            self.theta = theta
            
            # the calculations below will fail if the agent has not moved
            if x == self.x and y == self.y:
                self.dx = 0
                self.dy = 0
                return
                
            # dx and dy are all in the agents current frame of reference
            world_dx = self.x - x   # swapped due to mismatch in world coord frame             
            world_dy = y - self.y
            
            # the hypotenus of the triangle between the agents previous and current posistion
            
            h = math.sqrt(world_dx**2 + world_dy**2)
            theta_tilda = math.atan2(world_dy, world_dx)
            theta_prime = math.pi - theta_tilda - theta
            # theta_prime = theta - theta_tilda this should geometically be correct but there is something odd about world coordinate orgin
            
            self.dx = h*math.sin(theta_prime)
            self.dy = h*math.cos(theta_prime)
            # changes in x and y are all relative
            self.x = x
            self.y = y
            self.theta = theta
            
        @staticmethod
        def dist(x1,y1,x2,y2):
            return math.sqrt((x1-x2)**2 + (y1-y2)**2)    
        
        def has_looped(self, skip=16, thresh=64.0):
            if len(self.history) <=skip:
                return 0.0
            
            rest, p = self.history[:-skip], self.history[-1]
            
            x1,y1 = p
            for (x2,y2) in rest:
                if self.dist(x1,y1,x2,y2) < thresh:
                    return 1.0
            return 0.0            
     
    
    def __init__(self, params, idx=0, is_train=True, get_extra_info=False, use_shaping=False, fixed_scenario=False):
        
        self.depth_as_obs = params.depth_as_obs
        self.fixed_scenario = fixed_scenario
        self.is_train = is_train
        self.use_shaping = use_shaping
        self.game = self._create_game(params, idx, is_train, get_extra_info)
        self.predict_depth = params.predict_depth
        self.screen_width = params.screen_width
        self.screen_height = params.screen_height
        self.no_reward_average = params.no_reward_average        
        self.params = params
        self.ego_model = params.ego_model

        self.resize = params.resize
        self.frame_skip = params.frame_skip
        self.norm_obs = params.norm_obs
        
        self.action_map = self._gen_actions(self.game, params.limit_actions)
        self.num_actions = len(self.action_map)
        params.num_actions = self.num_actions
        self.player_info = self.PlayerInfo(
                self.game.get_game_variable(GameVariable.POSITION_X),
                self.game.get_game_variable(GameVariable.POSITION_Y),
                math.radians(self.game.get_game_variable(GameVariable.ANGLE)),
                fixed_origin=params.fixed_origin)
    
    def _create_game(self, params, idx, is_train, get_extra_info=False):
        game = DoomGame()
        self.idx = idx
        game.set_window_visible(params.show_window)
        game.set_sound_enabled(False)
        game.add_game_args("+vid_forcesurface 1")
        
        VALID_SCENARIOS = ['my_way_home.cfg',
                           'health_gathering.cfg',
                           'health_gathering_supreme.cfg',
                           'health_gathering_supreme_no_death_penalty.cfg',
                           'deadly_corridor.cfg',
                           'defend_the_center.cfg',
                           'defend_the_line.cfg',                      
                           'custom_maze_001.cfg',
                           'custom_maze_002.cfg',
                           'custom_maze_003.cfg',
                           'custom_mazes_005/train/maze_000.cfg',
                           'custom_mazes_005/train/maze_004.cfg',
                           'custom_mazes_005/valid/maze_000.cfg',
                           'long_term_base.cfg',
                           'scenario_x.cfg',
                           'scenario_cw2.cfg',
                           'scenario_2_item0.cfg',
                           'scenario_2_item1.cfg',
                           'scenario_2_item2.cfg',
                           'scenario_2_item3.cfg',
                           'scenario_3_item0.cfg',
                           'two_color_maze040.cfg',
                           'four_item_maze034.cfg',
                           'labyrinth_maze000.cfg',
                           'mino_maze000.cfg',
                           'labyrinth_maze11_000.cfg',
                           'mino_maze_simple.cfg']
        
        VALID_MULTI_SCENARIOS = ['maze_{:003}.cfg',
                                 'mino_maze{:003}.cfg',
                                 'labyrinth_maze{:003}.cfg',
                                 'indicator_maze{:003}.cfg',
                                 'two_item_maze{:003}.cfg',
                                 'six_item_maze{:003}.cfg',
                                 'four_item_maze{:003}.cfg',
                                 'eight_item_maze{:003}.cfg',
                                 'repeated_laby_maze{:003}.cfg',
                                 'two_color_maze{:003}.cfg']
        

        if params.scenario in VALID_SCENARIOS:
            game.load_config(params.scenario_dir + params.scenario)
        elif params.scenario in VALID_MULTI_SCENARIOS:
            assert params.multimaze
            if not is_train and params.test_scenario_dir:
                filename = params.test_scenario_dir + params.scenario.format(idx)
                #print('loading file', filename)
                game.load_config(filename)
            else:    
                filename = params.scenario_dir + params.scenario.format(idx)
                #print('loading file', filename)
                game.load_config(filename)
        elif params.scenario == 'curriculum':
            pass
        
        else:
            assert 0 , 'Invalid environment {}'.format(params.scenario)
            
        if params.screen_size == '320X180':
            game.set_screen_resolution(ScreenResolution.RES_320X180)
        else:
            assert 0 , 'Invalid screen_size {}'.format(params.screen_size)
        
        
        if (params.use_depth 
            or params.predict_depth 
            or params.ego_model 
            or params.depth_as_obs):
            
            game.set_depth_buffer_enabled(True)
            #self.game.set_labels_buffer_enabled(True)

        game.set_window_visible(params.show_window)
        game.set_sound_enabled(False)
        if params.show_window:
            game.set_mode(Mode.SPECTATOR)
            game.add_game_args("+freelook 1")
        
        # Player variables for prediction of position etc
        game.add_available_game_variable(GameVariable.POSITION_X)
        game.add_available_game_variable(GameVariable.POSITION_Y)
        game.add_available_game_variable(GameVariable.POSITION_Z)        
        game.add_available_game_variable(GameVariable.VELOCITY_X)
        game.add_available_game_variable(GameVariable.VELOCITY_Y)
        game.add_available_game_variable(GameVariable.VELOCITY_Z)  
        game.add_available_game_variable(GameVariable.ANGLE)       
        game.add_available_game_variable(GameVariable.PITCH)       
        game.add_available_game_variable(GameVariable.ROLL)       
        
        if get_extra_info:
            game.set_labels_buffer_enabled(True)
            game.set_automap_buffer_enabled(True)
            game.set_automap_mode(AutomapMode.OBJECTS)
            game.set_automap_rotate(True)
            game.set_automap_render_textures(False)
            game.set_depth_buffer_enabled(True)
        
        game.add_game_args("+vid_forcesurface 1")
        game.init() 
        
        if GameVariable.HEALTH in game.get_available_game_variables():
            self.previous_health = game.get_game_variable(GameVariable.HEALTH) 
            
        if self.use_shaping:
            self.shaping_reward = doom_fixed_to_double(game.get_game_variable(GameVariable.USER1))
            
        if params.disable_head_bob:
            game.send_game_command('movebob 0.0')
            
        return game
    
    def _gen_actions(self, game, limit_action_space):
        buttons = game.get_available_buttons()
        if buttons == [Button.TURN_LEFT, Button.TURN_RIGHT, Button.MOVE_FORWARD, Button.MOVE_BACKWARD]:
            if limit_action_space:
                feasible_actions = [[True, False, False,  False], # Left
                                    [False, True, False, False],  # Right
                                    [False, False, True, False],  # Forward
                                    [True, False, True, False],   # Left + Forward
                                    [False, True, True, False]]   # Right + forward
            else:
                feasible_actions = [[True, False, False,  False], # Left
                                    [False, True, False, False],  # Right
                                    [False, False, True, False],  # Forward
                                    [False, False, False, True],  # Backward
                                    [True, False, True, False],   # Left + Forward
                                    [True, False, False, True],   # Left + Backward
                                    [False, True, True, False],   # Right + forward
                                    [False, True, False, True]]   # Right + backward  
                

        
        else:
            feasible_actions = [list(l) for l in itertools.product([True, False], repeat=len(buttons))]
            print('Size of action space:', len(feasible_actions), self.params.num_actions)
            assert len(feasible_actions) == self.params.num_actions
            
            
        # else:
        #     assert 0, 'The feasible actions have not been implemented for this button combination'


        action_map = {i: act for i, act in enumerate(feasible_actions)}
        return action_map

    def load_new_scenario(self):
        # TODO
        self.game.set_doom_scenario_path(self.params.scenario_dir + self.params.scenario)
        self.game.new_episode()
        
    
    
    def reset(self, can_gen_rand=True):
        if ( not self.fixed_scenario and
             can_gen_rand and
             self.is_train and
             self.params.multimaze and 
             self.params.num_mazes_train > 16 and 
             random.randrange(0,10) == 0 ): # 1/10 chance to load a new map
            idx = random.randrange(0, self.params.num_mazes_train) 
            print('Creating new train maze with idx={}'.format(idx))
            self.game = self._create_game(self.params, idx, self.is_train)  
            
        if ( not self.fixed_scenario and
             can_gen_rand and
             not self.is_train and
             self.params.multimaze and 
             self.params.num_mazes_test > 1): # this is required during testing or the result is biased toward easier mazes
            idx = random.randrange(0, self.params.num_mazes_test) 
            print('Creating new test maze with idx={}'.format(idx))
            self.game = self._create_game(self.params, idx, self.is_train)   
    
        
    
        self.game.new_episode()
        self.player_info = self.PlayerInfo(
            self.game.get_game_variable(GameVariable.POSITION_X),
            self.game.get_game_variable(GameVariable.POSITION_Y),
            math.radians(self.game.get_game_variable(GameVariable.ANGLE)),
            fixed_origin=self.params.fixed_origin)
        
        if GameVariable.HEALTH in self.game.get_available_game_variables():
            self.previous_health = self.game.get_game_variable(GameVariable.HEALTH)
            
        if self.use_shaping:
            self.shaping_reward = doom_fixed_to_double(self.game.get_game_variable(GameVariable.USER1))       
        
        return self.get_observation()
    
    def is_episode_finished(self):
        return self.game.is_episode_finished()
    
    def get_observation(self):
        state = self.game.get_state()
        observation = state.screen_buffer

        if self.resize:
            # cv2 resize is 10x faster than skimage 1.37 ms -> 126 us
            observation = resize(
                    observation.transpose(1,2,0), 
                    (self.screen_width, self.screen_height), cv2.INTER_AREA
                 ).transpose(2,0,1)
            if self.depth_as_obs:
                depth = state.depth_buffer
                
                depth = resize(depth, 
                        (self.screen_width, self.screen_height), cv2.INTER_AREA
                      )  
                depth = np.expand_dims(depth, axis=0)
                observation = np.concatenate([observation, depth], axis=0)
            
        return self._normalize_observation(observation[:])
    
    def get_depth(self):
        assert self.predict_depth, 'Trying to predict depth but this option was not enabled in arguments'
        state = self.game.get_state()
        depth = state.depth_buffer
        return self._prepare_depth(depth)
    
    def get_ego_depth(self):
        assert self.ego_model, 'Trying to get depth without ego mode enabled'
        state = self.game.get_state()
        depth = state.depth_buffer
        return self._prepare_ego_depth(depth)
    
    def _prepare_depth(self, depth_buffer):
        """
            resize the depth buffer so it is the same size as the output of the models conv head
            discretize the values in range 0-7 so we can predict the depth in as a classification
        """
        # TODO: should I be dividing by 255.0, did I make a mistake?! 03/10/2018: this appears to be correct, although I have "improved" the normalization
        resized_depth = resize(depth_buffer, 
                               (self.screen_width//8, self.screen_height//8), 
                               cv2.INTER_AREA).astype(np.float32) * (1.0 /  255.0 )
        return np.clip(np.floor((1.4**(resized_depth)-1.0) * 50), 0.0, 7.0).astype(np.uint8)
    
    def _prepare_ego_depth(self, depth_buffer):
        """
            Resize the depth buffer and convert to real world coordinates for projective mapping
        """
        resized_depth = resize(depth_buffer, 
                               (self.screen_width // 8, self.screen_height // 8), 
                               cv2.INTER_AREA).astype(np.float32)
        return resized_depth * 7.5  # original =  7.4 + 7.0
    
    def _normalize_observation(self, observation):
        """
            Normalize the observation by making it in the range 0.0-1.0
            type conversion first is 2x faster
            multiplication is 4x faster than division
        """
        if self.norm_obs:
            return observation.astype(np.float32) * (1.0/255.0)
        else:
            return observation#.astype(np.float32)
    
    
    def make_action(self, action):
        """
            perform an action, includes an option to skip frames but repeat
            the same action.
            
        """     
        reward = self.game.make_action(self.action_map[action], self.frame_skip)
        
        if not self.use_shaping and self.is_train: # before I was using shaping I was comparing health
            reward += self._check_health()
        count = self.frame_skip

#        self.game.set_action(self.action_map[action])
#        self.game.advance_action(self.frame_skip)
#        reward = self.game.get_last_reward()
#        reward += self._check_health()
#        for skip in range(1, self.frame_skip):
#            if self.is_episode_finished(): 
#                break
#            reward += self.game.make_action(self.action_map[action])
#            reward += self._check_health()
#            count += 1.0
            
        if self.no_reward_average:
            count = 1.0

        if self.use_shaping and self.is_train:
            current_shaping_reward = doom_fixed_to_double(self.game.get_game_variable(GameVariable.USER1))
            diff = current_shaping_reward - self.shaping_reward
            reward += diff

            self.shaping_reward += diff
            
        return reward / count
    
    def step(self, action):
        reward = self.make_action(action)
        done = self.is_episode_finished()
        if done:
            obs = self.reset()
        else:
            new_x = self.game.get_game_variable(GameVariable.POSITION_X)
            new_y = self.game.get_game_variable(GameVariable.POSITION_Y)
            new_theta = self.game.get_game_variable(GameVariable.ANGLE)
            #print(new_x, new_y, new_theta)
            self.player_info.update(new_x, new_y, math.radians(new_theta))
            
            obs = self.get_observation()
            
        return obs, reward, done, None
    
    
    def _check_health(self):
        """
            Modification to reward function in order to reward the act of finding a health pack
        
        """
        health_reward = 0.0
    
        if GameVariable.HEALTH not in self.game.get_available_game_variables():
            self.previous_health = self.game.get_game_variable(GameVariable.HEALTH)
            return health_reward
        
        if self.game.get_game_variable(GameVariable.HEALTH) > self.previous_health:
            #print('found healthkit')
            health_reward = 1.0
            
        self.previous_health = self.game.get_game_variable(GameVariable.HEALTH)
        return health_reward
    
    def get_total_reward(self):
        return self.game.get_total_reward()
    
    def get_player_position(self):
        return self.player_info.x, self.player_info.y, self.player_info.theta

    def get_player_deltas(self):
        return self.player_info.dx, self.player_info.dy, self.player_info.dtheta

    def get_player_origins(self):
        return self.player_info.origin_x, self.player_info.origin_y
    
    def get_player_pos_delta_origin(self):
        if self.params.pos_as_obs:
            return ((self.player_info.x - self.player_info.origin_x)/128, 
                    (self.player_info.y - self.player_info.origin_y)/128, 
                    self.player_info.theta - self.player_info.starting_theta,
                    self.player_info.dx, self.player_info.dy, self.player_info.dtheta,
                    self.player_info.origin_x, self.player_info.origin_y)
        else:
            return (self.player_info.x, self.player_info.y, self.player_info.theta,
                    self.player_info.dx, self.player_info.dy, self.player_info.dtheta,
                    self.player_info.origin_x, self.player_info.origin_y)
        
    def get_loop(self):
        return self.player_info.has_looped()
  
def test():
     
    def simulate_rollout(env):
        from random import choice
        buffer = []
        env.reset()
        k = 0
        while not env.is_episode_finished():
            k += 1    
            obs = env.get_observation()
            buffer.append(obs)
    
            # Makes a random action and save the reward.
            reward = env.make_action(choice(list(range(env.num_actions))))   
        print('Game finished in {} steps'.format(k))
        print('Total rewards = {}'.format( env.get_total_reward()))
        return k, buffer        
   
    # =============================================================================
    #   Test the environment
    # =============================================================================
    
    from arguments import parse_game_args 
    params = parse_game_args()
    env = DoomEnvironment(params)
    print(env.num_actions)
    print(env.game.get_available_buttons())
    print(len(env.action_map))
    print(env.game.get_screen_height(), env.game.get_screen_width())

    print(env.get_observation().shape)

    import matplotlib.pyplot as plt
    
    plt.imshow(env.get_observation().transpose(1,2,0))
    plt.figure()
    plt.imshow(env.get_observation().transpose(1,2,0))
    
    env.decimate = False
    
    def resize_obs(observation):
        observation = observation.transpose(1,2,0)
        observation = resize(observation, (observation.shape[0]/2, observation.shape[1]/2))
        observation = observation.transpose(2,0,1)
        return observation
    
    
    data = env.get_observation().transpose(1,2,0)
    from skimage.transform import rescale, resize, downscale_local_mean

    data_resized = resize(data, (data.shape[0] / 2, data.shape[1] / 2))    
    
    plt.figure()
    plt.imshow(data_resized)    
    
    obs = env.get_observation()
    obs_rs = resize_obs(obs)
    
    assert 0
    for action in env.action_map.keys():
        reward = env.make_action(action)
        print(reward, env.is_episode_finished())
        
    for i in range(100):
        k, b = simulate_rollout(env)
        
  
    
    print(env.game.get_available_game_variables())
    print(env.game.get_game_variable(GameVariable.HEALTH))
    
   
def test_label_buffer():
    import matplotlib.pyplot as plt      
    import random
    from doom_rdqn.arguments import parse_game_args 
    params = parse_game_args()
    params.decimate = False
    env = DoomEnvironment(params)
    for i in range(10):
        env.make_action(random.choice(list(range(8))))
    
    
    
    
    state = env.game.get_state()
    labels_buffer = state.labels_buffer
    label = state.labels
    
    plt.subplot(1,2,1)
    plt.imshow(env.get_observation().transpose(1,2,0))
    plt.subplot(1,2,2)
    plt.imshow(labels_buffer)
    plt.figure()
    plt.imshow(resize(labels_buffer, (56,32), cv2.INTER_AREA))
    
    plt.figure()
    plt.imshow(resize(env.get_observation().transpose(1,2,0), (112,64), cv2.INTER_AREA))    
    
    
    data = env.get_observation()
    def resize_test(image):
        return resize(image.transpose(1,2,0), (112,64)).transpose(2,0,1)    



def load_scenario(env):
    env.load_new_scenario()
  
if __name__  == '__main__':
    import matplotlib.pyplot as plt      
    import random
    from doom_a2c.arguments import parse_game_args 
    params = parse_game_args()
    env = DoomEnvironment(params)
    
    
    obs = env.reset()
    print(obs.shape)
    for i in range(100):
        obs, _, _,_,  = env.step(3)
    print(obs.shape)


    rgb = obs[:3].transpose(1,2,0)
    
    depth = obs[3]
    plt.subplot(1,2,1)
    plt.imshow(rgb)
    plt.subplot(1,2,2)
    plt.imshow(depth)    
    
    pdo = env.get_player_pos_delta_origin()
