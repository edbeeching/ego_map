#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 11:18:31 2018

@author: anonymous

This code builds upon work found here:
    https://github.com/ikostrikov/pytorch-a2c-ppo-acktr
    

"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from logger import Logger
from doom_a2c.multi_env import MultiEnvsMP, MultiEnvsMPPipes
from doom_a2c.arguments import parse_game_args
from doom_a2c.rollout_buffer import RolloutStorage
from doom_a2c.models import CNNPolicy, EgoMap0_Policy
from doom_a2c.neural_map import NeuralMapPolicy
from doom_evaluation_multi import Evaluator


def train():
    # define params
    params = parse_game_args()
    print(params.num_actions)
    logger = Logger(params)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_updates = int(
        params.num_frames) // params.num_steps // params.num_environments

    # environments
    if params.use_pipes:
        envs = MultiEnvsMPPipes(
            params.simulator, params.num_environments, 1, params)
    else:
        envs = MultiEnvsMP(
            params.simulator, params.num_environments, 1, params)
    obs_shape = envs.obs_shape
    obs_shape = (obs_shape[0] * params.num_stack, *obs_shape[1:])

    evaluator = Evaluator(params)
    print('creating model')
    if params.ego_model:
        actor_critic = EgoMap0_Policy(
            obs_shape[0], obs_shape, params).to(device)

    elif params.neural_map:
        actor_critic = NeuralMapPolicy(
            obs_shape[0], obs_shape, params).to(device)
    else:
        actor_critic = CNNPolicy(obs_shape[0], obs_shape, params)

    if params.conv_head_checkpoint:
        print('loading conv head model')
        print(params.conv_head_checkpoint)
        assert os.path.isfile(
            params.conv_head_checkpoint), 'The pretrained model could not be loaded'

        checkpoint = torch.load(
            params.conv_head_checkpoint, map_location=lambda storage, loc: storage)

        old_model = CNNPolicy(obs_shape[0], obs_shape, params).cpu()

        old_model.load_state_dict(checkpoint['model'])

        actor_critic.load_conv_head(old_model)

    actor_critic.to(device)

    print('model created')
    start_j = 0
    if params.reload_model:
        checkpoint_idx = params.reload_model.split(',')[1]
        checkpoint_filename = '{}models/checkpoint_{}.pth.tar'.format(
            params.output_dir, checkpoint_idx)
        assert os.path.isfile(
            checkpoint_filename), 'The model could not be found {}'.format(checkpoint_filename)
        logger.write('Loading model{}'.format(checkpoint_filename))

        if device == 'cuda':  # The checkpoint will try to load onto the GPU storage unless specified
            checkpoint = torch.load(checkpoint_filename)
        else:
            checkpoint = torch.load(
                checkpoint_filename, map_location=lambda storage, loc: storage)
        actor_critic.load_state_dict(checkpoint['model'])

        start_j = (int(checkpoint_idx) // params.num_steps //
                   params.num_environments) + 1

    print('creating optimizer')
    optimizer = optim.RMSprop([p for p in actor_critic.parameters() if p.requires_grad],
                              params.learning_rate,
                              eps=params.eps,
                              alpha=params.alpha,
                              momentum=params.momentum)

    if params.reload_model:
        optimizer.load_state_dict(checkpoint['optimizer'])

    rollouts = RolloutStorage(params.num_steps,
                              params.num_environments,
                              obs_shape,
                              actor_critic.state_size,
                              params)

    current_obs = torch.zeros(params.num_environments, *obs_shape)

    # For Frame stacking
    def update_current_obs(obs):
        shape_dim0 = envs.obs_shape[0]
        obs = torch.from_numpy(obs).float()
        if params.num_stack > 1:
            current_obs[:, :-shape_dim0] = current_obs[:, shape_dim0:]
        current_obs[:, -shape_dim0:] = obs

    print('getting first obs')
    obs = envs.reset()
    print('update current obs')
    update_current_obs(obs)

    rollouts.observations[0].copy_(current_obs)

    if params.loop_detect:
        print('getting loops')
        loops = torch.from_numpy(envs.get_loops())
        rollouts.loops[0].copy_(loops)

    if params.neural_map:
        pos_deltas_origins = torch.from_numpy(envs.get_pos_deltas_origins())
        rollouts.pos_deltas_origins[0].copy_(pos_deltas_origins)

    if params.ego_model:
        ego_depths = torch.from_numpy(
            envs.get_ego_depths(trim=(not params.new_padding)))
        print('ego depths size', ego_depths.size())
        pos_deltas_origins = torch.from_numpy(envs.get_pos_deltas_origins())
        rollouts.ego_depths[0].copy_(ego_depths)
        rollouts.pos_deltas_origins[0].copy_(pos_deltas_origins)

    if params.pos_as_obs:
        pos_deltas_origins = torch.from_numpy(envs.get_pos_deltas_origins())
        rollouts.pos_deltas_origins[0].copy_(pos_deltas_origins)

    # These variables are used to compute average rewards for all processes.
    episode_rewards = torch.zeros([params.num_environments, 1])
    final_rewards = torch.zeros([params.num_environments, 1])

    current_obs = current_obs.to(device)
    rollouts.set_device(device)

    print('Starting training loop')
    start = time.time()
    print(num_updates)

    for j in range(start_j, num_updates):
        # STARTING no grad scope
        with torch.no_grad():
            if j % params.eval_freq == 0 and not params.skip_eval:
                print('Evaluating model')
                if params.simulator == 'doom':
                    actor_critic.eval()
                    total_num_steps = (
                        j + 1) * params.num_environments * params.num_steps
                    evaluator.evaluate(
                        actor_critic, params, logger, j, total_num_steps, params.eval_games)
                    actor_critic.train()

            # =============================================================================
            # Take steps in the environment
            # =============================================================================
            for step in range(params.num_steps):
                # Sample actions
                kwargs = {}

                if params.stacked_gru:
                    kwargs['states2'] = rollouts.states2[step]

                if params.neural_map:
                    kwargs['ego_states'] = rollouts.ego_states[step]
                    kwargs['pos_deltas_origins'] = rollouts.pos_deltas_origins[step]

                if params.ego_model:
                    kwargs['ego_states'] = rollouts.ego_states[step]
                    kwargs['ego_depths'] = rollouts.ego_depths[step]
                    kwargs['pos_deltas_origins'] = rollouts.pos_deltas_origins[step]

                    # pdo are already included as part of ego model

                if params.pos_as_obs:
                    kwargs['pos_deltas_origins'] = rollouts.pos_deltas_origins[step]

                result = actor_critic.act(
                    rollouts.observations[step],
                    rollouts.states[step],
                    rollouts.masks[step],
                    **kwargs)

                values = result['values']
                actions = result['actions']
                action_log_probs = result['action_log_probs']
                states = result['states']
                states2 = result.get('states2', None)
                ego_states = result.get('ego_states', None)

                cpu_actions = actions.squeeze(1).cpu().numpy()

                kwargs = {'ego_states': ego_states,
                          'states2': states2,
                          'prev_obs': torch.from_numpy(obs.copy())}
                # Obser reward and next obs
                obs, reward, done, info = envs.step(cpu_actions)

                if params.loop_detect:
                    kwargs['loops'] = torch.from_numpy(envs.get_loops())

                if params.predict_depth:
                    kwargs['depths'] = torch.from_numpy(envs.get_depths())

                if params.neural_map:
                    kwargs['pos_deltas_origins'] = torch.from_numpy(
                        envs.get_pos_deltas_origins())

                if params.ego_model:
                    kwargs['ego_depths'] = torch.from_numpy(
                        envs.get_ego_depths(trim=(not params.new_padding)))
                    kwargs['pos_deltas_origins'] = torch.from_numpy(
                        envs.get_pos_deltas_origins())

                if params.pos_as_obs:
                    kwargs['pos_deltas_origins'] = torch.from_numpy(
                        envs.get_pos_deltas_origins())

                reward = torch.from_numpy(
                    np.expand_dims(np.stack(reward), 1)).float()
                episode_rewards += reward

                # If done then clean the history of observations.
                masks = torch.FloatTensor(
                    [[0.0] if done_ else [1.0] for done_ in done])

                final_rewards *= masks
                final_rewards += (1 - masks) * episode_rewards
                episode_rewards *= masks

                masks = masks.to(device)

                if current_obs.dim() == 4:
                    current_obs *= masks.unsqueeze(2).unsqueeze(2)
                else:
                    current_obs *= masks

                update_current_obs(obs)

                rollouts.insert(step, current_obs, states, actions,
                                action_log_probs, values, reward,
                                masks,  **kwargs)

            # =============================================================================
            # Compute discounted returns, re-step through the environment
            # =============================================================================
            kwargs = {}

            if params.stacked_gru:
                kwargs['states2'] = rollouts.states2[-1]

            if params.neural_map:
                kwargs['ego_states'] = rollouts.ego_states[-1]
                kwargs['pos_deltas_origins'] = rollouts.pos_deltas_origins[-1]

            if params.ego_model:
                kwargs['ego_states'] = rollouts.ego_states[-1]
                kwargs['ego_depths'] = rollouts.ego_depths[-1]
                kwargs['pos_deltas_origins'] = rollouts.pos_deltas_origins[-1]

            if params.pos_as_obs:
                kwargs['pos_deltas_origins'] = rollouts.pos_deltas_origins[-1]

            next_value = actor_critic(rollouts.observations[-1],
                                      rollouts.states[-1],
                                      rollouts.masks[-1],
                                      **kwargs)['values']

            rollouts.compute_returns(
                next_value, params.use_gae, params.gamma, params.tau)

        # FINISHED no grad scope
        kwargs = {}

        if params.stacked_gru:
            kwargs['states2'] = rollouts.states2[0].view(
                -1, actor_critic.state_size)

        if params.neural_map:
            kwargs['ego_states'] = rollouts.ego_states[0]
            kwargs['pos_deltas_origins'] = rollouts.pos_deltas_origins[:-1]

        if params.ego_model:
            kwargs['ego_states'] = rollouts.ego_states[0]
            kwargs['ego_depths'] = rollouts.ego_depths[:-1]
            kwargs['pos_deltas_origins'] = rollouts.pos_deltas_origins[:-1]

        if params.pos_as_obs:
            kwargs['pos_deltas_origins'] = rollouts.pos_deltas_origins[:-1]

        model_output = actor_critic.evaluate_actions(rollouts.observations[:-1].view(-1, *obs_shape),
                                                     rollouts.states[0].view(
                                                         -1, actor_critic.state_size),
                                                     rollouts.masks[:-
                                                                    1].view(-1, 1),
                                                     rollouts.actions.view(-1, 1),
                                                     params.predict_depth,
                                                     **kwargs)

        values = model_output['values']
        action_log_probs = model_output['action_log_probs']
        dist_entropy = model_output['dist_entropy']
        depth_preds = model_output.get('depth_preds', None)

        #action_probs = model_output.get('action_probs', None)

        values = values.view(params.num_steps, params.num_environments, 1)
        action_log_probs = action_log_probs.view(
            params.num_steps, params.num_environments, 1)
        advantages = rollouts.returns[:-1] - values

        value_loss = advantages.pow(2).mean()
        action_loss = -(advantages.detach() * action_log_probs).mean()

        optimizer.zero_grad()

        loss = value_loss * params.value_loss_coef + \
            action_loss - dist_entropy * params.entropy_coef

        if params.loop_detect:
            loop_preds = model_output['loop_preds'].view(-1, 1)
            assert loop_preds is not None
            loop_targets = rollouts.loops[:-1].view(-1, 1)

            loop_loss = F.binary_cross_entropy(loop_preds, loop_targets)
            loss += params.loop_detect_lambda*loop_loss

        if params.action_prediction:
            action_preds = model_output.get('action_preds', None)
            assert action_preds is not None
            # compute and add the action loss
            cnn_output_size = 4*10

            if params.action_kl:
                #action_probs = action_probs.repeat(1, 1, cnn_output_size).detach()
                #cnn_action_probs = F.log_softmax(action_preds, dim=1)
                pass  # TODO
            else:
                actions = rollouts.actions
                actions_rep = actions.repeat(
                    1, 1, cnn_output_size).view(-1, 1, 4, 10)
                advantages_rep = advantages.detach().repeat(1, 1, cnn_output_size).view(-1, 1)

                cnn_action_probs = F.log_softmax(
                    action_preds, dim=1).gather(1, actions_rep).view(-1, 1)
                action_pred_loss = -(advantages_rep*cnn_action_probs).mean()

                loss += params.action_lambda*action_pred_loss

        if params.predict_depth:
            depth_preds = model_output.get('depth_preds', None)
            assert depth_preds is not None
            depth_preds = depth_preds.permute(
                0, 2, 3, 1).contiguous().view(-1, 8)
            depth_targets = rollouts.depths[:-1]

            depth_loss = F.cross_entropy(depth_preds, depth_targets.view(-1))
            loss = loss + params.depth_coef * depth_loss
        loss.backward()
        nn.utils.clip_grad_norm(
            actor_critic.parameters(), params.max_grad_norm)

        optimizer.step()
        rollouts.after_update()

        if j % params.model_save_rate == 0:
            total_num_steps = (j + 1) * \
                params.num_environments * params.num_steps
            checkpoint = {'step': step,
                          'params': params,
                          'model': actor_critic.state_dict(),
                          'optimizer': optimizer.state_dict()}

            filepath = logger.output_dir + 'models/'

            torch.save(checkpoint, '{}checkpoint_{:00000000010}.pth.tar'.format(
                filepath, total_num_steps))

        if j % params.log_interval == 0:
            end = time.time()
            total_num_steps = (j + 1) * \
                params.num_environments * params.num_steps
            save_num_steps = (start_j) * \
                params.num_environments * params.num_steps
            logger.write("Updates {}, num timesteps {}, FPS {}, mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}, entropy {:.5f}, value loss {:.5f}, policy loss {:.5f}".
                         format(j, total_num_steps,
                                int((total_num_steps - save_num_steps) /
                                    (end - start)),
                                final_rewards.mean(),
                                final_rewards.median(),
                                final_rewards.min(),
                                final_rewards.max(), dist_entropy.item(),
                                value_loss.item(), action_loss.item()))

            if params.ego_model and not params.conv_head_checkpoint:
                conv_weight_norm = actor_critic.conv_head[0].weight.norm() + \
                    actor_critic.conv_head[2].weight.norm() +  \
                    actor_critic.conv_head[4].weight.norm()

                conv_grad_norm = actor_critic.conv_head[0].weight.grad.norm() + \
                    actor_critic.conv_head[2].weight.grad.norm() +  \
                    actor_critic.conv_head[4].weight.grad.norm()

                if params.ego_skip_global:
                    ego_weight_norm = 0.0
                    ego_grad_norm = 0.0

                else:
                    ego_weight_norm = actor_critic.ego_head[0][0].weight.norm() + \
                        actor_critic.ego_head[0][2].weight.norm() +  \
                        actor_critic.ego_head[0][4].weight.norm()

                    ego_grad_norm = actor_critic.ego_head[0][0].weight.grad.norm() + \
                        actor_critic.ego_head[0][2].weight.grad.norm() +  \
                        actor_critic.ego_head[0][4].weight.grad.norm()

                logger.write('Conv: weight norm: {:1.5f}\tGrad norm: {:0.8f}\tEgo weight norm: {:1.5f}\tgrad norm: {:0.8f}'.format(
                    conv_weight_norm, conv_grad_norm, ego_weight_norm, ego_grad_norm))

    evaluator.cancel()
    envs.cancel()


if __name__ == "__main__":
    train()
