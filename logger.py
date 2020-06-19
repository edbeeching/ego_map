#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 10:04:41 2018

@author: anonymous
"""

import os
import datetime
import numpy as np
import visdom

def running_average(data, window=7):
    """
        Create running average of data

    """
    alpha = 2 /(window + 1.0)
    alpha_rev = 1-alpha
    n = data.shape[0]

    pows = alpha_rev**(np.arange(n+1))

    scale_arr = 1/pows[:-1]
    offset = data[0]*pows[1:]
    pw0 = alpha*alpha_rev**(n-1)

    mult = data*pw0*scale_arr
    cumsums = mult.cumsum()
    out = offset + cumsums*scale_arr[::-1]
    return out

class Logger(object):
    def __init__(self, params):
        
        self.params = params
        assert os.path.exists(params.scenario_dir)
        #assert os.path.exists(params.scenario_dir + params.scenario)
        if params.reload_model:
            
            results_dir = params.reload_model.split(',')[0]
            self.log_name = results_dir + '_log.txt'
            start_iter = params.reload_model.split(',')[1]
            params.output_dir = '/home/{}/tmp/results/{}_rl/{}/'.format(params.user_dir, params.simulator, results_dir)
            assert os.path.exists(params.output_dir), 'Trying to reload model but base output dir does not exist' 
            assert os.path.exists(params.output_dir + 'models/'), 'Trying to reload model but model output dir does not exist' 
            assert os.path.exists(params.output_dir + 'evaluations/'), 'Trying to reload model but eval output dir does not exist' 
            print(params.output_dir)
            self.output_dir = params.output_dir
            # Print to log file that training has resumed
            with open(self.output_dir + 'log.txt', 'a') as log:
                log.write('========== Resuming training iter: {} ========== \n'.format(int(start_iter)))
            
        else:
            now = datetime.datetime.now()
            print(now)    
        
            results_dir = '{}_{}_{:02}_{:02}_{:02}_{:02}_{:02}'.format(params.job_id, 
                                                              params.test_name,
                                                              now.year,
                                                              now.month, 
                                                              now.day, 
                                                              now.hour, 
                                                              now.minute)        
            self.log_name = results_dir + '_log.txt'
            params.output_dir = '/home/{}/tmp/results/{}_rl/{}/'.format(params.user_dir, params.simulator, results_dir)
            assert not os.path.exists(params.output_dir), 'The output directory already exists'
            self.output_dir = params.output_dir
            
            # create output directories  
            self.ensure_dir(self.output_dir)
            print('Created log output directory {}'.format(self.output_dir))
            
            #create model and eval dirs
            output_directories = ['models/', 'evaluations/']
            for directory in output_directories:
                self.ensure_dir(self.output_dir + directory)
             # Write params to a file
            with open(self.output_dir + 'log_params.txt', 'w') as params_log:
                for param, val in sorted(vars(params).items()):
                    params_log.write('{} : {} \n'.format(param, val))
            
            # Create log file
            with open(self.output_dir + self.log_name, 'w') as log:
                log.write('========== Training Log file ========== \n')
                
            if params.use_visdom:
                self.vis = visdom.Visdom(server=params.visdom_ip)
                self.vis_iters = []
                self.vis_scores = []
                
            
            
    def update_plot(self, iteration):
        smoothed_results = running_average(np.array(self.vis_scores))

        self.vis.line(X=np.stack([np.array(self.vis_iters),
                                   np.array(self.vis_iters)], axis=0).T,
                      Y=np.stack([np.array(self.vis_scores),
                                  smoothed_results], axis=0).T,
                      win=self.params.job_id*10 +1,
                      opts={'title':'test:{}_jobid:{}_it:{}_eval_scores'.format(
                              self.params.job_id, self.params.test_name, iteration)})
    
    def add_video(self, filename, best_agent):
        if best_agent:
            self.vis.video(videofile=filename, win=self.params.job_id*10 +2,
                           opts={'title':'test:{}_jobid:{}_it:{}_eval_video_best'.format(
                           self.params.job_id, self.params.test_name, self.vis_iters[-1])})
        else:
            self.vis.video(videofile=filename, win=self.params.job_id*10 +3,
                           opts={'title':'test:{}_jobid:{}_it:{}_eval_video_worse'.format(
                           self.params.job_id, self.params.test_name, self.vis_iters[-1])})

    def ensure_dir(self, file_path):
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):            
            os.makedirs(directory)  
            
    def write(self, text):
        now = datetime.datetime.now()
        print(now, ':', text)
        with open(self.output_dir + self.log_name, 'a') as log:
            log.write('{} :  {}\n'.format(now, text))
            
    def get_eval_output(self):
        return self.output_dir + 'evaluations/'
        
        
if __name__ =='__main__':
    
    from doom_a2c.arguments import parse_game_args
    params = parse_game_args()


    logger = Logger(params)
    logger.write('Epoch: 0 , loss = 10')
    
    params.logger = logger
    
    params.logger.write('test')    
