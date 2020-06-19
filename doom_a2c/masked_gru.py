#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 16:38:50 2018

@author: anonymous
"""


import torch
from torch import nn

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

class MaskedGRU(nn.Module):
    """
        A masked GRU which allows for sequentiall episodes
        e.g
        obserations [x1,x2,x3,x4,x1,x2,x3 ...]
        masks       [ 1, 1, 1, 1, 0, 1, 1 ...]
        
        masks ensure the hidden states are zeros between episodes
        
        
        Thanks to GitHub ikostrikov for the general implementation 
        which I have turned into a module
    
    """
    
    
    def __init__(self, num_in, num_out):
        super(MaskedGRU, self).__init__()
        self.gru = nn.GRU(num_in, num_out)
        
    
    def forward(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            # sequence of length one
            x, hxs = self.gru(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
        else:
            # more than one sequence
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0) \
                            .any(dim=-1)
                            .nonzero()
                            .squeeze()
                            .cpu())


            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]


            hxs = hxs.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                rnn_scores, hxs = self.gru(
                    x[start_idx:end_idx],
                    hxs * masks[start_idx].view(1, -1, 1)
                )

                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)
            hxs = hxs.squeeze(0)
        return x, hxs
    
    
    def reset_parameters(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                print('weight', name)
                torch.nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                print('bias')
                param.data.fill_(0)
    
if __name__ == '__main__':
    
    
    model = MaskedGRU(16,32)
    
    model.reset_parameters()
















    
