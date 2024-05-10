""" Utility classes and functions related to MoCL (NAACL 2024).
Copyright (c) 2024 Robert Bosch GmbH

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.
You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import os
import copy
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter


class TaskKeyEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.seed = config.seed
        
        self.task_list = config.task_list
        self.num_tasks = len(self.task_list)
        self.task_count = 0
        self.steps = {t_id: 0 for t_id in range(self.num_tasks)}
        self.steps_val = {t_id: 0 for t_id in range(self.num_tasks)}
        self.steps_final = {t_id: 0 for t_id in range(self.num_tasks)}
        
        self._initialize_keys()
        
        self.log_weights = True
        tb_log_dir = os.path.join(self.config.output_dir, f"tensorboard_seed{self.seed}")
        if not os.path.exists(tb_log_dir):
            os.makedirs(tb_log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=tb_log_dir)
        

    def forward(self, x_query, adapter_name, train=True, final=False):
        task_id = self.task_list.index(adapter_name)
        
        if train and not final: # training
            self.steps[task_id] += 1
        elif not train and not final: # validation
            self.steps_val[task_id] += 1
        else: # final evaluation
            self.steps_final[task_id] += 1
            
        batch_size = x_query.shape[0]
        loss = torch.tensor(0., requires_grad=True).cuda()
        
        K = self.keys[:task_id+1] # if not final and xx else self.keys 
        cos_sim = self._query_key_match(x_query, K)

        w = nn.functional.softmax(cos_sim*self.config.softmax_match_scale, dim=-1) if not self.config.direct_compose else cos_sim
        
        if train:
            loss, ortho_loss_key, matching_loss = self._add_kp_losses(K, w, cos_sim, loss, batch_size, task_id, x_query)
        
        if not train and not final: # during inference along with training
            w_avg = torch.mean(w.detach().clone(), dim=0)
            if self.attn_weights[task_id] is None:
                self.attn_weights[task_id] = w_avg
            else:
                self.attn_weights[task_id] = ((self.attn_weights[task_id] * self.steps_val[task_id]) + w_avg) / (self.steps_val[task_id]+1)
        
        self._log_weights(w, task_id, train, final)
                    
        if train:
            self._log_losses(task_id, ortho_loss_key, matching_loss)
            
        return w, loss
    
                
    def _initialize_keys(self):
        self.attn_weights = [None for _ in range(self.num_tasks)]
        self.keys = nn.Parameter(torch.randn((self.num_tasks, self.config.key_dim), requires_grad=True, device='cuda'))
            
        self.keys_mask = [False for _ in range(self.num_tasks)]
        
        if self.config.key_init_func == 'uniform':
            nn.init.uniform_(self.keys)
        
        elif self.config.key_init_func == 'orthogonal':
            self.keys = self.gram_schmidt(self.keys)
 
        self.ortho_loss_key = self.config.ortho_loss_key if self.config.key_init_func == 'orthogonal' else False
        self.ortho_loss_coeff = self.config.ortho_loss_coeff
        
    
    def _query_key_match(self, Q, K):
        n_k = nn.functional.normalize(K, dim=-1)
        n_q = nn.functional.normalize(Q, dim=-1).detach()
        
        assert n_k.dim() == n_q.dim() == 2
        cos_sim = torch.einsum('bj,kj->bk', n_q, n_k)

        for i in range(cos_sim.shape[-1]-1):
            if self.keys_mask[i]:
                cos_sim = cos_sim.clone()
                cos_sim[:, i] = -1e8
        return cos_sim
            
    
    def _add_kp_losses(self, K, w, cos_sim, loss, batch_size, task_id, x_query):
        ortho_loss_key = 0, 0
        if self.ortho_loss_key:
            ortho_loss_key = self.ortho_loss_coeff * self.ortho_penalty(K)
            loss += ortho_loss_key
            
        matching_loss = 0
        if self.config.matching_loss:
            # expect the (correct) weight digit = 1
            matching_loss = (1.0-w[:, -1]).mean() * self.config.matching_loss_coeff
            loss += matching_loss 
            
        if self.config.matching_loss_v2:
            # expect the (correct) query-key similarity = 1
            matching_loss = (1.0-cos_sim[:, -1]).mean() * self.config.matching_loss_coeff
            loss += matching_loss 
            
        if self.config.matching_loss_cls:
            # consider it as a classification task; gt_w is the ground truth one-hot vector
            gt_w = torch.zeros_like(w)
            gt_w[:, task_id] = 1.
            gt_w = gt_w[:, :task_id+1]
            
            matching_loss = torch.abs(gt_w - w).mean() * self.config.matching_loss_coeff
            loss += matching_loss 
                
        if self.config.matching_loss_cls_all:
            # consider it as a classification task, consider all keys (also include keys of future tasks)
            # gt_w is the ground truth one-hot vector
            gt_w = torch.zeros_like(w)
            gt_w[:, task_id] = 1.
            
            matching_loss = torch.abs(gt_w - w).mean() * self.config.matching_loss_coeff
            loss += matching_loss 
            
        return loss, ortho_loss_key, matching_loss
    
    
    def _add_ortho_loss(self, keys, prompts, loss):
        ortho_loss_key, ortho_loss_prompt = 0, 0
        if self.ortho_loss_key:
            ortho_loss_key = self.ortho_loss_coeff * self.ortho_penalty(keys)
            loss += ortho_loss_key

        return loss, ortho_loss_key, ortho_loss_prompt
    
    
    def _log_weights(self, w, task_id, train, final):
        if self.log_weights:
            # In total 4 modes: 1-train, 2-inference, 3-final-local, 4-final-global
            w_avg = torch.mean(w.detach().clone(), dim=0)
            
            for t in range(w_avg.shape[-1]):
                if train and not final:
                    step = self.steps[task_id]
                    log_name = f'training_weight_{task_id}/{t}'
                elif not train and not final:
                    step = self.steps_val[task_id]
                    log_name = f'validation_weight_{task_id}/{t}'
                elif final:
                    step = self.steps_final[task_id]
                    log_name = f'final_weight_{task_id}/{t}'


                self.writer.add_scalar(log_name, w_avg[t], step)


    def _log_losses(self, task_id, ortho_loss_key, matching_loss):
        if self.ortho_loss_key:
            self.writer.add_scalar(f'ortho_loss/task_{task_id}', ortho_loss_key.cpu(), self.steps[task_id])
        if self.config.matching_loss or self.config.matching_loss_v2 or self.config.matching_loss_cls or self.config.matching_loss_cls_all:
            self.writer.add_scalar(f'matching_loss/task_{task_id}', matching_loss.cpu(), self.steps[task_id])

    
    # code for this function is from:
    # https://github.com/legendongary/pytorch-gram-schmidt/blob/master/gram_schmidt.py
    def gram_schmidt(self, vv):

        def projection(u, v):
            denominator = (u * u).sum()

            if denominator < 1e-8:
                return None
            else:
                return (v * u).sum() / denominator * u
        
        # check if the tensor > 2D and flatten the last dimensions if necessary
        is_nd = len(vv.shape) > 2
        if is_nd:
            shape_nd = copy.deepcopy(vv.shape)
            vv = vv.view(vv.shape[0], -1)
            
        # swap rows and columns
        vv = vv.T

        # process matrix size
        nk = vv.size(1)
        uu = torch.zeros_like(vv, device=vv.device)
        
        s = self.task_count
        if s>0:
            uu[:, 0:s] = vv[:, 0:s].clone()
            
        for k in range(s, nk):
            redo = True
            while redo:
                redo = False
                vk = torch.randn_like(vv[:,k]).to(vv.device)
                # vk = vv[:, k].clone()
                uk = 0
                for j in range(0, k):
                    if not redo:
                        uj = uu[:, j].clone()
                        proj = projection(uj, vk)
                        if proj is None:
                            redo = True
                            print('restarting!')
                        else:
                            uk = uk + proj
                if not redo:
                    uu[:, k] = vk - uk
                    
        for k in range(s, nk):
            uk = uu[:, k].clone()
            uu[:, k] = uk / uk.norm()
            
        # undo swapping of rows and columns
        uu = uu.T 
        
        # return from 2D
        if is_nd:
            uu = uu.view(shape_nd)
        
        return nn.Parameter(uu) 
    
    
    def ortho_penalty(self, t):
        # return ((t @t.T - torch.eye(t.shape[0]).cuda())**2).mean()
        return ((t @t.T - torch.eye(t.shape[0]).cuda()).abs()).mean()

    
    def process_task_count(self):
        self.task_count += 1
        
        # project untrained keys to ensure orthogonality
        if self.config.key_init_func == 'orthogonal':
            self.keys = self.gram_schmidt(self.keys)