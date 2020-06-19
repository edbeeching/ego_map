#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 09:53:21 2018

@author: anonymous
"""
import matplotlib.pyplot as plt
import numpy as np
from time import sleep
from vizdoom import ScreenResolution, Mode, GameVariable, DoomGame, doom_fixed_to_double
from environments import DoomEnvironment
from doom_a2c.arguments import parse_game_args

#####################################################################
# This script presents SPECTATOR mode. In SPECTATOR mode you play and
# your agent can learn from it.
# Configuration is loaded from "../../scenarios/<SCENARIO_NAME>.cfg" file.
# 
# To see the scenario description go to "../../scenarios/README.md"
#####################################################################

params = parse_game_args()
params.scenario = 'mino_maze_simple.cfg'
params.limit_actions = True
params.show_window = True
params.no_reward_average = True
env = DoomEnvironment(params, use_shaping=True)


env.reset()
episodes = 10
inv_action_map = {tuple(v):k for k,v in env.action_map.items()}
for i in range(episodes):
    print("Episode #" + str(i + 1))

    rewards = []
    done = False
    while not done:

        action = env.game.get_last_action()
        action_bool = tuple(bool(a) for a in action)
        
        if action_bool not in inv_action_map:
            action_ind = 0
        else:
            action_ind = inv_action_map[action_bool]
            
        obs, reward, done, info = env.step(action_ind)
        rewards.append(reward)     
        
        sleep(4/30.0)     
        


    print("Episode finished!")
    print(rewards)
    #print("Total reward:", game.get_total_reward())
    print("************************")
    sleep(2.0)

#game.close()
