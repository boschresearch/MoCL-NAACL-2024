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

import torch
import torch.nn as nn

from model.mtl_prefix_encoder import PrefixContinualMTLEncoder, PrefixQKVEncoder
from utils.utilities import mahalanobis


class T5PrefixContinualForConditionalGeneration(nn.Module):
    def __init__(self, training_args, model_args, model, tokenizer, task_list, task2target_len):
        super(T5PrefixContinualForConditionalGeneration, self).__init__()
        self.model = model
        self.config = model.config
        self._add_config(training_args, model_args)
        self.tokenizer = tokenizer
        self.task_list = task_list
        self.num_tasks = len(task_list)
        self.task2target_len = task2target_len
        self.device = training_args.device
        
        for name, param in self.model.named_parameters():
            param.requires_grad = False
        
        self.query_encoder = self.model.encoder
        # query and keys should have the same dimension
        self.config.key_dim = model.config.hidden_size
        self.query_size = model.config.hidden_size
        self.task_infer_acc = []
        
        if self.config.compose_prompts:
            self.embed_prototypes = [None for _ in range(self.num_tasks)]
            self.prefix_encoder = PrefixQKVEncoder(self.config)
        else:
            self.prefix_encoder = PrefixContinualMTLEncoder(self.config)
        
        if self.config.task_identify_epi:
            self.task_embeds = [[] for _ in range(self.num_tasks)]
            self.task_labels = [[] for _ in range(self.num_tasks)]
            
            self.task_means_over_classes = nn.ParameterList()
            # share covariance acrosss all tasks
            self.accumulate_shared_covs = nn.Parameter(torch.zeros(
                self.query_size, self.query_size), requires_grad=False)
            self.cov_inv = nn.Parameter(torch.ones(
                self.query_size, self.query_size), requires_grad=False)
        
        t5_param = 0
        for name, param in self.model.named_parameters():
            t5_param += param.numel()
        all_param = 0
        for name, param in self.named_parameters():
            all_param += param.numel()
        total_param = all_param - t5_param
        print('all param is {}M'.format(all_param/(1024*1024)))
        print('trainable param is {}M'.format(total_param/(1024*1024)))
            
    
    def get_prompt(self, batch_size, task_id, x_query=None, train=False, final=False, prompt_select_mode='local_compose', task_id_list=None, id_pred_on=False):
        input0 = x_query if self.config.compose_prompts else batch_size 
        outputs = self.prefix_encoder(input0, task_id, train, final, prompt_select_mode, task_id_list, id_pred_on)
        
        if self.config.compose_prompts:
            past_key_values, match_loss = outputs
        else:
            # vanilla CL & ProgressivePrompt & EPI without composition
            past_key_values =  outputs
            match_loss = torch.tensor(0., requires_grad=True).cuda()            
        
        past_prompt = []
        for i, key_val in enumerate(past_key_values):
            temp = {}
            temp['encoder_prompt'] = {
                'prev_key': key_val[0].contiguous(),
                'prev_value': key_val[1].contiguous(),
                'prev_key_padding_mask': torch.zeros(key_val.shape[1], key_val.shape[3]).to(key_val.device).bool()
            }
            temp['decoder_prompt'] = {
                'prev_key': key_val[0].contiguous(),
                'prev_value': key_val[1].contiguous(),
                'prev_key_padding_mask': torch.zeros(key_val.shape[1], key_val.shape[3]).to(key_val.device).bool()
            }
            temp['cross_attention_prompt'] = {
                'prev_key': key_val[0].contiguous(),
                'prev_value': key_val[1].contiguous(),
                'prev_key_padding_mask': torch.zeros(key_val.shape[1], key_val.shape[3]).to(key_val.device).bool()
            }
            past_prompt.append(temp)
            
        return past_prompt, match_loss
        
        
    def forward(
        self,
        batch,
        task_id,
        train=False,
        final=False, # final evaluation 
        prompt_select_mode='local_compose',
        id_pred_on=False,
    ):
        batch = {k: batch[k].to(self.device) for k in batch}
        
        attention_mask = batch["source_mask"]
        labels = batch["target_ids"]
        labels[labels[:, :] == self.tokenizer.pad_token_id] = -100
        
        inputs_embeds = self.model.encoder.embed_tokens(batch["source_ids"])
        batch_size = inputs_embeds.shape[0]
        
        x_query = None
        task_id_list=None
        if self.config.compose_prompts or self.config.task_identify_epi:
            x_embed = self.query_encoder(
                inputs_embeds = inputs_embeds,
                attention_mask = attention_mask,
                head_mask=None,  
                output_attentions=None,  
                output_hidden_states=None, 
                return_dict=None,  
            ).last_hidden_state
            
            x_query = self.process_query(x_embed, attention_mask, train, task_id, labels)

            if final:
                if id_pred_on: # pred task ids for prompt selection
                    if not (self.config.task_identify_epi or self.config.classifier_match_embed or self.config.classifier_match_key):
                        self.config.classifier_match_key = True
                    task_id_list, task_id_prob = self._get_pred_ids(x_query, task_id, final)

        past_prompt, match_loss = self.get_prompt(batch_size, task_id, x_query, train, final, prompt_select_mode, task_id_list, id_pred_on)
        
        if train:
            outputs = self.model(
                input_ids=batch["source_ids"],
                attention_mask=attention_mask,
                labels=labels,
                past_prompt=past_prompt,
            )
            
            total_loss = outputs['loss'] + match_loss
            
            try:
                self.prefix_encoder.writer.add_scalar(f'task_loss/task_{task_id}', outputs['loss'].cpu(), self.prefix_encoder.steps[task_id])
                self.prefix_encoder.writer.add_scalar(f'total_loss/task_{task_id}', total_loss.cpu(), self.prefix_encoder.steps[task_id])
            except:
                pass
            
            # return outputs['loss'] + match_loss.item(), None
            return total_loss, None
        else:
            outputs = self.model.generate(
                input_ids=batch["source_ids"],
                attention_mask=attention_mask,
                past_prompt=past_prompt,
                max_length=self.task2target_len[self.task_list[task_id]],
            ) 
        
            return None, outputs
        
        
    def process_query(self, embed, attention_mask, train, task_id, labels=None):
        # Using 'avg_all_embed' by default
        x_query = embed.mean(dim=1)
        
        # store embedding prototypes if required
        if train and self.config.classifier_match_embed:
            if self.embed_prototypes[task_id] is not None:
                steps = self.prefix_encoder.steps[task_id]
                self.embed_prototypes[task_id] = (self.embed_prototypes[task_id]*(steps-1) + x_query.mean(dim=0))/steps
            else:
                self.embed_prototypes[task_id] = x_query.mean(dim=0)
        
        if train and self.config.task_identify_epi and self.task_embeds[task_id]==[]:
            # only add once is enough
            self.task_embeds[task_id].extend(x_query.tolist())
            self.task_labels[task_id].extend(labels.tolist())
            
        return x_query
    
        
    def _get_pred_ids(self, x_query, task_id, final, log_id='cil'):
        # TODO: rename, this is not for classifier match (but for general task matching/inference)
        if self.config.classifier_match_embed:
            embed_prototypes = torch.stack(self.embed_prototypes)
            cos_sim = torch.cosine_similarity(x_query.unsqueeze(1), embed_prototypes.unsqueeze(0), dim=-1) # [bs, num_tasks]
            prob = nn.functional.softmax(cos_sim*self.config.softmax_match_scale, dim=-1) # if not self.config.direct_compose else cos_sim; comment this because for the prediction, softmax is required and scale is necessary in this case.
            pred_ids = torch.argmax(prob, dim=-1) # [bs]
            
        elif self.config.classifier_match_key:
            K = self.prefix_encoder.keys
            cos_sim = torch.cosine_similarity(x_query.unsqueeze(1), K.unsqueeze(0), dim=-1) # [bs, num_tasks]
            prob = nn.functional.softmax(cos_sim*self.config.softmax_match_scale, dim=-1) # if not self.config.direct_compose else cos_sim
            pred_ids = torch.argmax(prob, dim=-1) # [bs]
            
        elif self.config.task_identify_epi: # epi has the lowest priority
            pred_ids, prob = self._get_epi_ids(x_query)
            log_id='epi'
                    
        tg_ids = torch.ones_like(pred_ids).to(pred_ids.device) * task_id
        accuracy = (pred_ids == tg_ids).float()
        step = self.prefix_encoder.steps_final[task_id]
        self.prefix_encoder.writer.add_scalar(f'key_match_acc/{log_id}_task_{task_id}', accuracy.mean(), step)
        self.task_infer_acc.append(accuracy.cpu().mean().item())
        return pred_ids, prob


    def _get_epi_ids(self, query):
        scores_over_tasks = []
        for mean_over_classes in self.task_means_over_classes:
            num_labels = mean_over_classes.shape[0]
            score_over_classes = []
            for l in range(num_labels):
                score = mahalanobis(query, mean_over_classes[l], self.cov_inv, norm=2)
                score_over_classes.append(score)
            
            # [num_labels, bs]
            score_over_classes = torch.stack(score_over_classes)
            score, _ = score_over_classes.min(dim=0)
            
            scores_over_tasks.append(score)
        
        # [num_tasks, bs]
        scores_over_tasks = torch.stack(scores_over_tasks, dim=0)
        _, ids = torch.min(scores_over_tasks, dim=0)
        
        prob_over_tasks = (scores_over_tasks*(-1)).T.softmax(dim=-1)
        
        return ids, prob_over_tasks

    def _add_config(self, training_args, model_args):
        self.config.seed = training_args.seed
        self.config.output_dir = training_args.output_dir
        
        for arg in dir(model_args):
            if not arg.startswith("__") and not callable(getattr(model_args, arg)):
                setattr(self.config, arg, getattr(model_args, arg))