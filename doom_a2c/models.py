#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 10:53:06 2018

@author: anonymous
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('..')

if __name__ == '__main__':
    from distributions import Categorical
    from ego_map import EgoMap
else:
    from .distributions import Categorical    
    from .ego_map import EgoMap

# A temporary solution from the master branch.
# https://github.com/pytorch/pytorch/blob/7752fe5d4e50052b3b0bbc9109e599f8157febc0/torch/nn/init.py#L312
# Remove after the next version of PyTorch gets release.
def orthogonal(tensor, gain=1):
    if tensor.ndimension() < 2:
        raise ValueError("Only tensors with 2 or more dimensions are supported")

    rows = tensor.size(0)
    cols = tensor[0].numel()
    flattened = torch.Tensor(rows, cols).normal_(0, 1)

    if rows < cols:
        flattened.t_()

    # Compute the qr factorization
    q, r = torch.qr(flattened)
    # Make Q uniform according to https://arxiv.org/pdf/math-ph/0609050.pdf
    d = torch.diag(r, 0)
    ph = d.sign()
    q *= ph.expand_as(q)

    if rows < cols:
        q.t_()

    tensor.view_as(q).copy_(q)
    tensor.mul_(gain)
    return tensor

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        orthogonal(m.weight.data)
        print(m.__class__.__name__)
        print(m.weight.size())
        if m.bias is not None:
            m.bias.data.fill_(0)


class Lin_View(nn.Module):
	def __init__(self):
		super(Lin_View, self).__init__()
        
	def forward(self, x):
		return x.view(x.size()[0], -1)


class FFPolicy(nn.Module):
    def __init__(self):
        super(FFPolicy, self).__init__()

    def forward(self, inputs, states, masks):
        raise NotImplementedError

    def act(self, inputs, states, masks, 
            deterministic=False, **kwargs):

        result = self(inputs, states, masks, **kwargs)
        
        x = result['x']
        actions = self.dist.sample(x, deterministic=deterministic)
        action_log_probs, dist_entropy, action_probs = self.dist.logprobs_and_entropy(x, actions)

        del result['x']      
        result['actions'] = actions
        result['dist_entropy'] = dist_entropy
        result['action_log_probs'] = action_log_probs
        result['action_probs'] = action_probs  
        
        return result

    def evaluate_actions(self, inputs, states, 
                         masks, actions, pred_depths=False, **kwargs):
        if pred_depths:
            result = self(inputs, states, masks, pred_depths=pred_depths, **kwargs)
            
            x = result['x']
            action_log_probs, dist_entropy, action_probs = self.dist.logprobs_and_entropy(x, actions)
            
            del result['x']      
            result['actions'] = actions
            result['dist_entropy'] = dist_entropy
            result['action_log_probs'] = action_log_probs
            result['action_probs'] = action_probs            
        
            return result
        else:   
            result = self(inputs, states, masks, **kwargs)
            
            x = result['x']
            action_log_probs, dist_entropy, action_probs = self.dist.logprobs_and_entropy(x, actions)
            del result['x']      
            
            result['actions'] = actions
            result['dist_entropy'] = dist_entropy
            result['action_log_probs'] = action_log_probs
            result['action_probs'] = action_probs             
            
            return result
    
    def get_action_value_and_probs(self, inputs, states, masks, 
                                   deterministic=False, **kwargs):
        
        result = self(inputs, states, masks, **kwargs)
    
        x = result['x']
        actions = self.dist.sample(x, deterministic=deterministic)
        
        result['actions'] = actions 
        result['action_softmax'] = F.softmax(self.dist(x),dim=1)
        del result['x']      
        return result


class CNNPolicy(FFPolicy):
    def __init__(self, num_inputs, input_shape, params):
        super(CNNPolicy, self).__init__()

        if params.pretrained_vae:
            class Args:
                pass
            args=Args()
            args.hidden2=params.hidden_size
            self.vae = VAE2(args)
        else:
            self.conv_head = nn.Sequential(nn.Conv2d(num_inputs, params.conv1_size, 8, stride=4),
                                           nn.ReLU(True),
                                           nn.Conv2d(params.conv1_size, params.conv2_size, 4, stride=2),
                                           nn.ReLU(True),
                                           nn.Conv2d(params.conv2_size, params.conv3_size, 3, stride=1),
                                           nn.ReLU(True))
            if params.action_prediction:
                # an additional conv head for action prediction
                self.cnn_action_head = nn.Conv2d(params.conv3_size, params.num_actions,1,1)
    
    
            if params.predict_depth:
                # predict depth with a 1x1 conv
                self.depth_head = nn.Conv2d(params.conv3_size, 8, 1, 1)
    
            conv_input = torch.Tensor(torch.randn((1,) + input_shape))
            print(conv_input.size(), self.conv_head(conv_input).size(), self.conv_head(conv_input).size())
            self.conv_out_size = self.conv_head(conv_input).nelement()    
        self.hidden_size = params.hidden_size
        
        if params.skip_fc:
            if params.recurrent_policy:
                assert params.use_lstm == False, 'Cannot have both GRU and LSTM!'
                #self.gru = MaskedGRU(self.conv_out_size, self.hidden_size)
                
                self.gru = nn.GRUCell(self.conv_out_size, self.hidden_size)

                if params.stacked_gru:
                    #self.gru2 = MaskedGRU(self.hidden_size, self.hidden_size)
                    self.gru2 = nn.GRUCell(self.hidden_size, self.hidden_size)

            if params.use_lstm:
                self.lstm = nn.LSTMCell(self.conv_out_size, self.hidden_size)
        else:
            if not params.pretrained_vae:
                self.linear1 = nn.Linear(self.conv_out_size, self.hidden_size)

            if params.recurrent_policy:
                assert params.use_lstm == False, 'Cannot have both GRU and LSTM!'
                #self.gru = MaskedGRU(self.hidden_size, self.hidden_size)
                if params.pos_as_obs:
                    self.gru = nn.GRUCell(self.hidden_size + 4, self.hidden_size) # x, y sin(orientation), cos(orientation)
                else:
                    self.gru = nn.GRUCell(self.hidden_size, self.hidden_size)

                if params.stacked_gru:
                    #self.gru2 = MaskedGRU(self.hidden_size, self.hidden_size)
                    self.gru2 = nn.GRUCell(self.hidden_size, self.hidden_size)
            if params.use_lstm:
                self.lstm = nn.LSTMCell(self.hidden_size, self.hidden_size)
        
        if params.gru_skip:
            self.critic_linear = nn.Linear(self.hidden_size*2, 1)
            self.dist = Categorical(self.hidden_size*2, params.num_actions)
        else:
            self.critic_linear = nn.Linear(self.hidden_size, 1)
            self.dist = Categorical(self.hidden_size, params.num_actions)
        
        if params.loop_detect:
            self.loop_linear = nn.Linear(self.hidden_size, 1)

        self.params = params
        self.train()
        self.reset_parameters()
        
        if params.pretrained_vae:
            enc_checkpoint = torch.load(params.pretrained_vae, map_location=lambda storage, loc: storage) 
            self.vae.load_state_dict(enc_checkpoint['model'])       
            self.vae.eval()
        
        
    @property
    def state_size(self):
        if hasattr(self, 'gru') or hasattr(self, 'lstm'):
            return self.hidden_size
        else:
            return 1

    def load_conv_head(self, old_model):
        for i in range(0, 6, 2):
            self.conv_head[i].weight.data = old_model.conv_head[i].weight.data.clone()
            self.conv_head[i].bias.data = old_model.conv_head[i].bias.data.clone()
            self.conv_head[i].weight.requires_grad = False
            self.conv_head[i].bias.requires_grad = False
            
    def load_linear_layer(self, old_model):
        self.linear1.weight.data = old_model.linear1.weight.data.clone()
        self.linear1.bias.data = old_model.linear1.bias.data.clone()
        self.linear1.weight.requires_grad = False
        self.linear1.bias.requires_grad = False

    def reset_parameters(self):
        self.apply(weights_init)

        relu_gain = nn.init.calculate_gain('relu')
        if not self.params.pretrained_vae:
            for i in range(0, 6, 2):
                self.conv_head[i].weight.data.mul_(relu_gain)
            self.linear1.weight.data.mul_(relu_gain)
        if self.params.loop_detect:
            self.loop_linear.weight.data.mul_(relu_gain)

        if hasattr(self, 'gru'):
            #self.gru.reset_parameters()
            
            orthogonal(self.gru.weight_ih.data)
            orthogonal(self.gru.weight_hh.data)
            self.gru.bias_ih.data.fill_(0)
            self.gru.bias_hh.data.fill_(0)
            if self.params.learn_init_state:
                self.init_state = nn.Parameter(torch.randn(1, self.hidden_size) * 0.00) 
            if self.params.gru_forget_init:
                self.gru.bias_ih.data.uniform_(-self.params.gru_bias_range, self.params.gru_bias_range)
                self.gru.bias_hh.data.uniform_(-self.params.gru_bias_range, self.params.gru_bias_range)

        if hasattr(self, 'gru2'):
            #self.gru2.reset_parameters()
            
            orthogonal(self.gru2.weight_ih.data)
            orthogonal(self.gru2.weight_hh.data)
            self.gru2.bias_ih.data.fill_(0)
            self.gru2.bias_hh.data.fill_(0)
            if self.params.learn_init_state:
                self.init_state2 = nn.Parameter(torch.randn(1, self.hidden_size) * 0.00) 
            if self.params.gru_forget_init:
                self.gru2.bias_ih.data.uniform_(-self.params.gru_bias_range, self.params.gru_bias_range)
                self.gru2.bias_hh.data.uniform_(-self.params.gru_bias_range, self.params.gru_bias_range)
                
        if hasattr(self, 'lstm'):
            orthogonal(self.lstm.weight_ih.data)
            orthogonal(self.lstm.weight_hh.data)
            self.lstm.bias_ih.data.fill_(1)
            self.lstm.bias_hh.data.fill_(1)
            if self.params.gate_init:
                self.lstm.bias_ih.data.fill_(0)
                self.lstm.bias_hh.data.fill_(0)

        if self.dist.__class__.__name__ == "DiagGaussian":
            self.dist.fc_mean.weight.data.mul_(0.01)
            
    def pred_depth(self, inputs):   
        x = self.conv_head(inputs * (1.0/255.0))
        return self.depth_head(x)       
        
    def forward(self, inputs, states, masks, 
                pred_depth=False, pos_deltas_origins=None, **kwargs):
        
        depth_preds = None
        action_preds = None
        
        if self.params.pos_as_obs:
            assert pos_deltas_origins is not None
            
        
        if self.params.pretrained_vae:
            with torch.no_grad():
                self.vae.eval()
                x = self.vae.encode(inputs * (1.0/128.0))[0].detach()
                x = x.view(-1, self.params.hidden_size)
        else:
            x = self.conv_head(inputs * (1.0/self.params.image_scalar))
    
            if pred_depth:
                depth_preds = self.depth_head(x)
            
            if self.params.action_prediction:
                action_preds = self.cnn_action_head(x)

            x = x.view(-1, self.conv_out_size)
            if not self.params.skip_fc:
                x = self.linear1(x)
                x = F.relu(x)

        if self.params.gru_skip:
            rnn_input = x

        if hasattr(self, 'gru'):
            if hasattr(self,'gru2'):
                states2 = kwargs['states2']
                assert states2 is not None
            else:
                states2 = None
                
            # x, states = self.gru(x, states, masks)
            # if hasattr(self,'gru2'):
            #      x, states2 = self.gru2(x, states2, masks)  
            if inputs.size(0) == states.size(0):
                if self.params.pos_as_obs:
                    dxys, dthetas = pos_deltas_origins[:,3:5], pos_deltas_origins[:,5:6]
                    x = torch.cat([x, dxys, torch.sin(dthetas), torch.cos(dthetas)], dim=1)  
                x = states = self.gru(x, states*masks)
                if hasattr(self,'gru2'):
                    x = states2 = self.gru2(x, states2*masks)                
            else:
                x = x.view(-1, states.size(0), x.size(1))
                masks = masks.view(-1, states.size(0), 1)
                outputs = []
                
                for i in range(x.size(0)):
                    if hasattr(self, 'gru2'):
                        hx = states = self.gru(x[i], states * masks[i])
                        hx2 = states2 = self.gru2(hx, states2 * masks[i])
                        outputs.append(hx2)         
                    else:
                        inp = x[i]
                        if self.params.pos_as_obs:
                            dxys, dthetas = pos_deltas_origins[i, :,3:5], pos_deltas_origins[i, :,5:6]
                            inp = torch.cat([inp, dxys, torch.sin(dthetas), torch.cos(dthetas)], dim=1)  
                        hx = states = self.gru(inp, states * masks[i])
                        outputs.append(hx)
                        
                x = torch.cat(outputs, 0)                     
                    
        loop_preds = None
        if self.params.loop_detect:
            loop_preds = F.sigmoid(self.loop_linear(x))
        
        if self.params.gru_skip:
            x = torch.cat([rnn_input, x], 1)
        

        result = {'values': self.critic_linear(x), 
                  'x': x, 
                  'states': states, 
                  'states2': states2, 
                  'depth_preds': depth_preds, 
                  'action_preds': action_preds,
                  'loop_preds': loop_preds}

        return result

class EgoMap0_Policy(FFPolicy):
    def __init__(self, num_inputs, input_shape, params):
        super(EgoMap0_Policy, self).__init__()

        self.params = params
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
        self.ego_map = EgoMap(params)
        ac = 0 # addtional channel
        if params.ego_curiousity:
            ac = 1        
        if not params.ego_skip_global:    
            self.ego_head = nn.Sequential(nn.Conv2d(params.ego_num_chans + ac, params.ego_num_chans, 3, stride=1, padding=1),
                                           nn.ReLU(True),
                                           nn.Conv2d(params.ego_num_chans, params.ego_num_chans, 4, stride=2, padding=0),
                                           nn.ReLU(True),
                                           nn.Conv2d(params.ego_num_chans, params.ego_num_chans, 4, stride=2, padding=0),
                                           nn.ReLU())
    
            ego_head_input = torch.Tensor(torch.randn((1,) + (params.ego_num_chans + ac, 
                                                             2*params.ego_half_size, 
                                                             2*params.ego_half_size,)))
            ego_out_size = self.ego_head(ego_head_input).nelement()
        
            # This has to be defined as an extension as I do not know the size in advance
            if params.ego_use_tanh:
                ego_extension = nn.Sequential(Lin_View(),
                                  nn.Linear(ego_out_size, 256),
                                  nn.ReLU(True),
                                  nn.Linear(256, params.ego_hidden_size),
                                  nn.Tanh())
            else:
                ego_extension = nn.Sequential(Lin_View(),
                                  nn.Linear(ego_out_size, 256),
                                  nn.ReLU(True),
                                  nn.Linear(256, params.ego_hidden_size),
                                  nn.ReLU(True))
            
            self.ego_head = nn.Sequential(self.ego_head, ego_extension)
            ego_out_size = params.ego_hidden_size
            self.ego_out_size = ego_out_size  
             
        conv_input = torch.Tensor(torch.randn((1,) + input_shape))
        conv_out_size = self.conv_head(conv_input).nelement()
        
        if params.ego_curiousity:
            conv_out_size += 4*10
        
        self.conv_out_size = conv_out_size
        print('conv out size', self.conv_out_size)

        if params.ego_skip and not params.ego_skip_global and not params.merge_later:    
            self.linear1 = nn.Linear(ego_out_size + conv_out_size, params.hidden_size)   
        elif params.ego_skip and (params.merge_later or params.ego_skip_global):
            print('params.ego_skip and (params.merge_later or params.ego_skip_global)')
            self.linear1 = nn.Linear(conv_out_size, params.hidden_size)
        else:
            self.linear1 = nn.Linear(ego_out_size, params.hidden_size)

        if params.recurrent_policy:
            assert params.use_lstm == False, 'Cannot have both GRU and LSTM!'
            if params.merge_later and not params.ego_skip_global:
                self.gru = nn.GRUCell(params.hidden_size + ego_out_size, params.hidden_size)
            else:
                self.gru = nn.GRUCell(params.hidden_size, params.hidden_size)
                

        if params.ego_query:
            if params.ego_query_scalar:
                query_out_size = params.ego_num_chans + 1 + ac 
            else:
                query_out_size = params.ego_num_chans + ac 
            
            if params.ego_skip_global:
                self.query_head = nn.Linear(params.hidden_size, query_out_size)
            else:
                self.query_head = nn.Linear(ego_out_size + params.hidden_size, query_out_size)
                
            if params.query_position:
                self.critic_linear = nn.Linear(params.hidden_size + params.ego_num_chans + 2 + ac, 1)
                self.dist = Categorical(params.hidden_size + params.ego_num_chans + 2 + ac, params.num_actions)
            else:
                self.critic_linear = nn.Linear(params.hidden_size + params.ego_num_chans + ac, 1)
                self.dist = Categorical(params.hidden_size + params.ego_num_chans + ac, params.num_actions)
        else:
            self.critic_linear = nn.Linear(params.hidden_size, 1)
            self.dist = Categorical(params.hidden_size, params.num_actions)
            
        

        print(params.num_actions, ' actions')
        self.train()
        self.reset_parameters()
        
        if params.ego_curiousity:
            # Load the forward model
            class Args:
                pass

            args = Args()
            args.hidden2 = 128
            args.load_vae = ''
            args.shared_action_size = 32
            args.load_forward = ''
            save_name = params.forward_model_name
            model = ForwardModel(args)
            checkpoint = torch.load(save_name, map_location=lambda storage, loc: storage)
            model.load_state_dict(checkpoint['model'])
            model.eval()
            
            self.forward_model = model
        
    @property
    def state_size(self):
        if hasattr(self, 'gru') or hasattr(self, 'lstm'):
            return self.params.hidden_size
        else:
            return 1
        
    def load_conv_head(self, old_model):
        for i in range(0, 6, 2):
            self.conv_head[i].weight.data = old_model.conv_head[i].weight.data.clone()
            self.conv_head[i].bias.data = old_model.conv_head[i].bias.data.clone()
            self.conv_head[i].weight.requires_grad = False
            self.conv_head[i].bias.requires_grad = False
            
    def reset_parameters(self):
        self.apply(weights_init)

        relu_gain = nn.init.calculate_gain('relu')
        for i in range(0, 6, 2):
            self.conv_head[i].weight.data.mul_(relu_gain)
            
            if not self.params.ego_skip_global:
                self.ego_head[0][i].weight.data.mul_(relu_gain)
            
        if not self.params.ego_skip_global:
            for i in range(1, 5, 2):
                self.ego_head[1][i].weight.data.mul_(relu_gain)                                     
            
        self.linear1.weight.data.mul_(relu_gain)
 
    
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



    def forward(self, inputs, states, masks, 
                pred_depth=False, 
                ego_states=None, ego_depths=None, 
                pos_deltas_origins=None, states2=None,
                prev_obs=None, prev_actions=None, curiousity=None):
        

        assert ego_states is not None, 'Trying to apply ego update with states'
        assert ego_depths is not None, 'Trying to apply ego update with no depths'
        assert pos_deltas_origins is not None, 'Trying to apply ego update with no pdo'
        
        if self.params.ego_curiousity:
            assert prev_obs is not None
            assert prev_actions is not None
            if curiousity is None:
                curiousity = self.calculate_curiousity(inputs, prev_obs, pos_deltas_origins, prev_actions)
        

        conv_out = self.conv_head(inputs * (1.0/255.0))
        if self.params.ego_curiousity: # concat the curiousity vectors for ego reading
            assert conv_out.size(0) == curiousity.size(0)
            assert masks.size(0) == curiousity.size(0)
            conv_out = torch.cat([conv_out, curiousity*masks.view(-1,1,1,1)], 1)

        if inputs.size(0) == states.size(0):                
            xxs, yys, thetas = pos_deltas_origins[:,0], pos_deltas_origins[:,1], pos_deltas_origins[:,2]
            origin_x, origin_y = pos_deltas_origins[:,6], pos_deltas_origins[:,7]
            
            ego_states = self.ego_map.ego_mapper(conv_out, ego_depths, ego_states * masks.view(-1,1,1,1), xxs, yys, thetas, masks, origin_x, origin_y)
            
            x = ego_rots = self.ego_map.rotate_for_read(ego_states, xxs, yys, thetas, origin_x, origin_y)    
        else:
            x = conv_out.view(-1, states.size(0), * conv_out.size()[1:])
            
            masks = masks.view(-1, states.size(0), 1)

            read_outputs = []
            
            for i in range(x.size(0)):
                xxs, yys, thetas = pos_deltas_origins[i, :,0], pos_deltas_origins[i, :,1], pos_deltas_origins[i, :,2]
                origin_x, origin_y = pos_deltas_origins[i, :,6], pos_deltas_origins[i, :,7]
                ego_states = self.ego_map.ego_mapper(x[i], ego_depths[i], ego_states * masks[i].unsqueeze(-1).unsqueeze(-1), 
                                                     xxs, yys, thetas, masks[i], origin_x, origin_y)
                
                read = self.ego_map.rotate_for_read(ego_states, xxs, yys, thetas, origin_x, origin_y)
                read_outputs.append(read)   
            
            x = ego_rots = torch.cat(read_outputs, 0)
   
        if self.params.ego_skip_global: #do not include a global CNN read on egomap
            if self.params.skip_cnn_relu: # Relu was not applied earlier so apply now
                conv_out = F.relu(conv_out.view(-1, self.conv_out_size))   
            else:
                conv_out = conv_out.view(-1, self.conv_out_size)
            
            x = self.linear1(conv_out)
            x = F.relu(x)  
        else:
            x = ego_reads = self.ego_head(x).view(-1, self.ego_out_size)
                    
            if not self.params.ego_skip: #do not include skip connect 
                x = self.linear1(x)
                x = F.relu(x)             
            
            if self.params.ego_skip and not self.params.merge_later:
                if self.params.skip_cnn_relu: # Relu was not applied earlier so apply now
                    conv_out = F.relu(conv_out.view(-1, self.conv_out_size))   
                else:
                    conv_out = conv_out.view(-1, self.conv_out_size)
                x = torch.cat([x, conv_out], dim=1)
                x = self.linear1(x)
                x = F.relu(x)        
           
            
            if self.params.ego_skip and self.params.merge_later:
                if self.params.skip_cnn_relu: # Relu was not applied earlier so apply now
                    conv_out = F.relu(conv_out.view(-1, self.conv_out_size))   
                else:
                    conv_out = conv_out.view(-1, self.conv_out_size)
                y = self.linear1(conv_out)
                y = F.relu(y)
                x = torch.cat([x, y], dim=1)
            
            
        if hasattr(self, 'gru'):
            if inputs.size(0) == states.size(0):
                if self.params.learn_init_state:
                    x = states = self.gru(x, states * masks + (1-masks)*self.init_state.clone().repeat(states.size(0), 1))
                else:
                    x = states = self.gru(x, states * masks)
            else:
                
                x = x.view(-1, states.size(0), x.size(1))
                outputs = []
                for i in range(x.size(0)):
                    if self.params.learn_init_state:
                        hx = states = self.gru(x[i], states * masks[i] + (1-masks[i])*self.init_state.clone().repeat(states.size(0), 1))
                    else:
                        hx = states = self.gru(x[i], states * masks[i])
                    outputs.append(hx)
                x = torch.cat(outputs, 0)
        
        # note as gru and lstm gave comparable results, just gru is used
        if self.params.ego_query:
            if self.params.ego_skip_global:
                query_vectors = self.query_head(x)
            else:
                query_vectors = self.query_head(torch.cat([x, ego_reads], dim=1))
                
            if self.params.query_relu:
                query_vectors = F.relu(query_vectors)
                
            context_vectors = self.ego_map.query(query_vectors, ego_rots)
            
#            print('context norms')
#            print(context_vectors.norm(), x.norm())
            x = torch.cat([x, context_vectors], dim=1)

        result = {'values': self.critic_linear(x), 
                  'x': x, 
                  'states': states, 
                  'ego_states': ego_states,
                  'curiousity': curiousity}

        return result
          
    def calculate_curiousity(self, inputs, prev_obs, pos_deltas_origins, actions):
        with torch.no_grad():
            self.forward_model.eval()
            actions = actions.view(-1)
            # calulate
            # 1. VAE encoder + decode of current obs
            o2_hat, mus1, logvars1 = self.forward_model.vae(inputs/128.0)
            # 2. forward prediction of previous obs with actions and deltas
            o1_hat, mus0, logvars0 = self.forward_model.vae(prev_obs/128.0)
            deltas = pos_deltas_origins[:, 3:5]
            action_emb = self.forward_model.action_embedding(actions, deltas)
            # TODO: test logvar conversion
            
            p_in = torch.cat([action_emb, mus0, logvars0], 1)
            p_out = self.forward_model.p_mlp(p_in)
            mus_t1_hat, logvars_t1_hat = p_out[:,:self.forward_model.args.hidden2],  p_out[:,self.forward_model.args.hidden2:]
            z = self.forward_model.vae.reparametrize(mus_t1_hat, logvars_t1_hat)
            o2t1_hat = self.forward_model.vae.decode(z)        
        
            # TODO: compare with abs error
            error =  ((o2_hat.detach() - o2t1_hat.detach())**2).mean(dim=1, keepdim=True)
            error = F.interpolate(error, size=(8,14))
            if not self.params.new_padding:
                error = error[:,:,2:-2,2:-2]
        
        return error * (1.0/100.0) # normlizing to range of around 0-5  
        
        
if __name__ == '__main__':
    
    from doom_a2c.arguments import parse_game_args 
    params = parse_game_args()  
    params.num_actions = 5
    
    # ego_model = EgoMap0_Policy(3, (3, 64, 112), params)
    
    # neural_map_model = NeuralMapPolicy(3, (3, 64, 112), params)

    # ego_states =  torch.zeros(2, params.ego_num_chans, 
    #                            params.ego_half_size*2 -1, 
    #                            params.ego_half_size*2 -1)
    
    # query_vector = torch.randn(2,params.ego_num_chans)
    
    # result, scores = neural_map_model.query(query_vector, ego_states)
    
    # result
    # scores
    
    
    model = CNNPolicy(3, (3, 64, 112), params)
    
    example_input = torch.randn(1,3,64,112)
    
    out = model.conv_head[:4](example_input)
    
    print(out.size())
