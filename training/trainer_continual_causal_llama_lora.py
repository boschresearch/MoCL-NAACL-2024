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
import json
import math
import shutil
import numpy as np

import torch
import torch.nn as nn
from torch.optim import AdamW

from collections import defaultdict
from tqdm import tqdm

from transformers import GenerationConfig
from transformers.trainer_pt_utils import get_parameter_names
from transformers.optimization import get_scheduler
from training.early_stopping import EarlyStopping

from mpeft import get_peft_model
from mpeft.utils.save_and_load import set_peft_model_state_dict

from utils.compute_metrics import compute_metrics
from utils.utilities import mahalanobis


class ContinualTrainerMTL(nn.Module):
    def __init__(self,
                 args,
                 model,
                 query_encoder,
                 logger,
                 task_list,
                 label_list,
                 peft_config,
                 lora_save_dir,
                 early_stopping_patience=-1,
                 tokenizer=None,
                 max_target_length=20,
                 learning_rate_list=None):
        super(ContinualTrainerMTL, self).__init__()
        
        self.args = args
        self.seed = args.seed
        self.model = model.to(self.args.device)
        self.query_encoder = query_encoder.to(self.args.device)
        self.logger = logger
        self.task_list = task_list
        self.label_list = label_list
        self.num_tasks = len(task_list)
        self.peft_config = peft_config
        
        if self.model.config.task_identify_epi:
            # assert self.model.config.disentangle_modules
            self._initialize_epi_variables()
        
        self.num_train_epochs = math.ceil(args.num_train_epochs)
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping = EarlyStopping(
            save_path=os.path.join(args.output_dir, 'best_checkpoint'),
            logger=self.logger,
            patience = early_stopping_patience
        )
        self.tokenizer = tokenizer
        self.max_target_length=max_target_length
        self.metric = 'rougeL'
        
        try:
            self.learning_rate_list = [float(x) for x in learning_rate_list.split('_')]
        except:
            self.learning_rate_list = [self.args.learning_rate for _ in range(self.num_tasks)]

        
        self.logger.info(f"***********************************seed: {self.seed}***********************************")
        self.logger.info(f"***********************************lr: {self.learning_rate_list}***********************************")
        self.logger.info(f"***********************************lr_scheduler_type: {self.args.lr_scheduler_type}***********************************")

    
    def _prepare_optimizer(self, task_id=None):
        decay_parameters = get_parameter_names(self.model, [nn.LayerNorm])
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if n in decay_parameters],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if n not in decay_parameters],
                "weight_decay": 0.0,
            },
        ]
        
        # get optimizer class and kwargs
        learning_rate = self.learning_rate_list[task_id] if self.learning_rate_list is not None else self.args.learning_rate
        optimizer_kwargs = {"lr": learning_rate}
        
        adam_kwargs = {
            "betas": (self.args.adam_beta1, self.args.adam_beta2),
            "eps": self.args.adam_epsilon,
        }

        optimizer_cls = AdamW
        optimizer_kwargs.update(adam_kwargs)

        # Initilaize optimizer
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
    
    
    def _prepare_scheduler(self, num_training_steps, optimizer):
        self.lr_scheduler = get_scheduler(
                    self.args.lr_scheduler_type,
                    optimizer=self.optimizer,
                    num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                    num_training_steps=num_training_steps,
                )


    def _process_data(self, loader, mode):
        try:
            data_batch = next(loader[1])
        except:
            loader[1] = iter(loader[0])
            data_batch = next(loader[1])
        data_batch = {k: data_batch[k].to(self.args.device) for k in data_batch}
        
        data_batch["labels"] = data_batch["labels"].long()
        label = data_batch["labels"].to(self.args.device)
        if mode != 'train':
            label[label[:, :] == -100] = self.tokenizer.pad_token_id
            
        return data_batch, label

    
    def compute_loss(self, model, inputs, task, mode, query_embed=None, final=False):
        if mode == 'train':
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                loss_mask=inputs["loss_mask"],
                labels=inputs["labels"],
                active_adapter=task,
                query_embed=query_embed,
                disentangle_modules=self.model.config.disentangle_modules,
                train=True,
                final=False,
            )
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            return loss, outputs["logits"]
        else:
            gen_kwargs = {"max_new_tokens": inputs["max_new_token"]}
            gen_kwargs["attention_mask"] = inputs["attention_mask"]
            gen_kwargs["repetition_penalty"] = 1.0 
            gen_kwargs["eos_token_id"] = self.tokenizer.eos_token_id
            gen_kwargs["pad_token_id"] = self.tokenizer.pad_token_id
            
            generated_tokens = model.generate(
                inputs=inputs["input_ids"],
                active_adapter=task,
                query_embed=query_embed,
                disentangle_modules=self.model.config.disentangle_modules,
                generation_config=GenerationConfig(**gen_kwargs),
                train=False,
                final=final,
                )
            return generated_tokens
    
    
    def _prepare_dataloaders(self, dataloaders):
        loader = {}
        batch_num = []
        for task in dataloaders.keys():
            loader[task] = [dataloaders[task], iter(dataloaders[task])]
            batch_num.append(len(dataloaders[task]))
        return loader, batch_num
    
    
    def train(self, 
              train_dataloaders, 
              val_dataloaders,
              test_dataloaders,
              ):

        train_loader, train_batch = self._prepare_dataloaders(train_dataloaders)
        val_loader, val_batch = self._prepare_dataloaders(val_dataloaders)
        test_loader, test_batch = self._prepare_dataloaders(test_dataloaders)
        
        
        fwd_pass_results_metric = defaultdict(list)
        if self.args.do_train:

            for task_id, task in enumerate(self.task_list):
                if task_id == 0 or self.model.config.multi_peft_modules:
                    self.model = get_peft_model(self.model, self.peft_config, adapter_name=task)
                    self.model.print_trainable_parameters()
                    self.model.set_adapter(adapter_name=task)
                    
                num_train_batch = train_batch[task_id]
                num_training_steps = self.num_train_epochs * num_train_batch
                num_examples_dataset = train_batch[task_id] * self.args.per_device_train_batch_size
                num_examples_training = num_training_steps * self.args.per_device_train_batch_size
                len_dataloader = len(train_dataloaders[task])
                num_update_steps_per_epoch = len_dataloader // self.args.gradient_accumulation_steps
                num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
                num_training_steps_all_epochs = math.ceil(self.num_train_epochs * num_update_steps_per_epoch)
                
                self._prepare_optimizer(task_id)
                self._prepare_scheduler(num_training_steps_all_epochs, self.optimizer)

                self.model.epochs = self.num_train_epochs
                self.model.task_id = task_id
                self.early_stopping.reinit()
                eval_res, test_res = defaultdict(list), defaultdict(list)

                # Train!
                self.logger.info(f"***** Running training - task {task_id}: {task}*****")
                self.logger.info(f"  Num Epochs = {self.num_train_epochs}")
                self.logger.info(f"  Num examples datasets = {num_examples_dataset}")
                self.logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size}")
                self.logger.info(f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}")
                self.logger.info(f"  Total optimization steps = {num_training_steps_all_epochs}")
                self.logger.info(f"  Num examples training = {num_examples_training}")

                
                for epoch in range(self.num_train_epochs):
                    self.model.epoch = epoch
                    self.model.train()
                    train_preds, train_gts = [], []
                    
                    for batch_index in tqdm(range(num_train_batch), total=num_training_steps_all_epochs):
                        train_input, train_gt = self._process_data(train_loader[task], mode='train')
                        query_embed = self._get_query_embed(train_input, task, do_statistics=(epoch==0 and self.model.config.task_identify_epi))
                        train_loss, train_pred = self.compute_loss(self.model, train_input, task, mode='train', query_embed=query_embed)
                        train_loss = train_loss / self.args.gradient_accumulation_steps
                        
                        train_preds.append(train_pred)
                        train_gts.append(train_gt)
                        
                        self.model.zero_grad()
                        train_loss.backward()
                        if batch_index % self.args.gradient_accumulation_steps == 0:
                            self.optimizer.step()
                            self.lr_scheduler.step() 
                            # print(f"learning_rate: {self.lr_scheduler.get_last_lr()[0]}")
                            
                    eval_res_ = self.eval(val_loader[task], val_batch[task_id], task, mode='val')
                    for k, v in eval_res_.items():
                        eval_res[k].append(v)
                    
                    if epoch==0 and self.model.config.task_identify_epi:
                        self._statistics_task_mean_cov(task, task_id)
                    
                    self.clean_gpu_memory()
                    
                    best_epoch = epoch
                    es_task_name = task if self.model.config.multi_peft_modules else self.task_list[0]
                    if self.early_stopping_patience > 0 and (self.early_stopping(eval_res_[self.metric], self.model, es_task_name) or epoch == self.num_train_epochs-1):
                        best_metric_eval = self.early_stopping.best_score
                        best_epoch = eval_res[self.metric].index(best_metric_eval)
                        
                        self.logger.info(f"Early stop!!! Best epoch for [{task}]: {best_epoch}")
                        self.logger.info(f"Best evaluation {self.metric} [{task}]: {round(best_metric_eval, 4)}")
                    
                        # load weights of the best model
                        model_path = os.path.join(self.early_stopping.save_path, f'best_checkpoint_{task}.pth')
                        self.logger.info(f"Loading weights from the best model (best epoch {best_epoch})...")
                        set_peft_model_state_dict(self.model, self.early_stopping.best_model, adapter_name=es_task_name)
                        if task_id != len(self.task_list) - 1 and os.path.exists(model_path):
                            os.remove(model_path)
                            
                        best_res_test = self.eval(test_loader[task], test_batch[task_id], task, mode='test')
                        
                        self.logger.info(f"Best test {self.metric} [{task}]: {round(best_res_test[self.metric], 4)}")
                        for k, v in best_res_test.items():
                            fwd_pass_results_metric[k].append(v)
                        
                        break

                if self.model.config.multi_peft_modules and not self.model.config.disentangle_modules:
                    self.model.key_encoder.process_task_count()
                    
        if self.model.config.disentangle_modules:
            test_results_metric = fwd_pass_results_metric
        else:
            self.logger.info(f"------- Forward pass evaluation (seed-{self.seed}) -------")
            for k in fwd_pass_results_metric.keys():
                self._log_final_results(test_loader, fwd_pass_results_metric[k], key=f"{k}")
            
            test_results_metric = defaultdict(list)
            for t_id, t in enumerate(test_loader.keys()):
                self.logger.info(f"------- Final evaluation (seed-{self.seed}) on task [{t}] -------")
                test_res = self.eval(test_loader[t], test_batch[t_id], t, mode='test', final=True)
                for k, v in test_res.items():
                    test_results_metric[k].append(v)
                    
        for k in test_results_metric.keys():
            self._log_final_results(test_loader, test_results_metric[k], fwd_pass_results_metric[k], key=k)
        
        if self.model.config.task_identify_epi:
            epi_test_results_metric = defaultdict(list)
            for t_id, t in enumerate(test_loader.keys()):
                
                self.logger.info(f"------- Final EPI evaluation (seed-{self.seed}) on task [{t}] -------")
                test_res = self.epi_eval(test_loader[t], test_batch[t_id], t_id, mode='test')
                for k, v in test_res.items():
                    epi_test_results_metric[k].append(v)
                    
            for k in epi_test_results_metric.keys():
                self._log_final_results(test_loader, epi_test_results_metric[k], key=f"EPI-{k}")
                
            self._print_task_infer_acc()
        # remove checkpoints
        shutil.rmtree(self.early_stopping.save_path)


    def _get_query_embed(self, model_inputs, task=None, do_statistics=False):
        input_ids = model_inputs['input_ids']
        attention_mask = model_inputs['attention_mask']
        with torch.no_grad():
            hidden_states = self.query_encoder(
                input_ids,
                attention_mask=attention_mask,
            )[0]
        
        query_encoder_type = self.model.config.query_encoder_type
        if query_encoder_type == 'avg_all_embed':
            embed = hidden_states.mean(dim=1)
        elif query_encoder_type == 'avg_word_embed':
            embed = torch.sum(hidden_states * attention_mask.unsqueeze(-1), dim=1) / torch.sum(attention_mask, dim=1).unsqueeze(-1)
        
        if do_statistics:
            labels = model_inputs['target_ids']
            self.task_embeds[task].extend(embed.tolist())
            self.task_labels[task].extend(labels.tolist())
        
        return embed
    
    
    def eval(self, loader, batch, task, mode='test', final=False):        
        self.model.eval()
        results = defaultdict(list)
        with torch.no_grad():
            for batch_index in tqdm(range(batch)):
                input, _ = self._process_data(loader, mode)
                
                query_embed = self._get_query_embed(input)
                input["max_new_token"] = self.max_target_length
                generated_tokens = self.compute_loss(self.model, input, task, mode=mode, query_embed=query_embed, final=final)
                
                predictions = self._postprocess_generated_tokens(generated_tokens.cpu())
                references = self.tokenizer.batch_decode(input["targets"], skip_special_tokens=True, clean_up_tokenization_spaces=True)
                
                metrics = compute_metrics(predictions, references)
                for key, value in metrics.items():
                    results[key].append(value)
        
        if final:
            self.logger.info(f"***** Task {task} - Final evaluation: {mode} *****")
        else:
            self.logger.info(f"***** Task {task} - Epoch {self.model.epoch}: {mode} (seed {self.seed}) *****")
        
        for key, value in results.items(): 
            avg_value = round(sum(value)/len(value), 4)
            results[key] = avg_value
            self.logger.info(f"{key}: {avg_value}")
        
        return results
    
            
    def epi_eval(self, loader, batch, task_id, mode='test'):        
        self.model.eval()
        results = defaultdict(list)
        with torch.no_grad():
            for batch_index in tqdm(range(batch)):
                input, _ = self._process_data(loader, mode)
                
                query_embed = self._get_query_embed(input)
                pred_ids = self._get_epi_ids(query_embed)
                pred_id = torch.argmax(torch.bincount(pred_ids)).item()
                input["max_new_token"] = self.max_target_length
                generated_tokens = self.compute_loss(self.model, input, self.task_list[pred_id], mode=mode, query_embed=query_embed, final=True)
                
                predictions = self._postprocess_generated_tokens(generated_tokens.cpu())
                references = self.tokenizer.batch_decode(input["targets"], skip_special_tokens=True, clean_up_tokenization_spaces=True)
                
                metrics = compute_metrics(predictions, references)
                for key, value in metrics.items():
                    results[key].append(value)
                
                tg_ids = torch.ones_like(pred_ids).to(pred_ids.device) * task_id
                accuracy = (pred_ids == tg_ids).float()
                self.task_infer_acc[task_id].append(accuracy.cpu().mean().item())
                
        self.logger.info(f"***** Task {self.task_list[task_id]} - Final EPI evaluation: {mode} *****")
        
        for key, value in results.items(): 
            avg_value = round(sum(value)/len(value), 4)
            results[key] = avg_value
            self.logger.info(f"{key}: {avg_value}")
        
        return results
                    
        
    def _postprocess_generated_tokens(self, generated_tokens):
        generated_tokens = np.where(generated_tokens==-100, self.tokenizer.pad_token_id, generated_tokens)
        predictions = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        
        answer_prefix = "Answer:"
        final_predictions = []
        for pred in predictions:
            if answer_prefix in pred:
                splits = pred.split(answer_prefix)
                final_predictions.append(splits[-1].strip())
            else:
                final_predictions.append('')
        
        return final_predictions

    
    def _log_final_results(self, test_loader, test_results, fwd_pass_results=None, key='acc'):
        self.logger.info(f"***********************************seed: {self.seed} - {key}***********************************")
        
        self.logger.info(f"Test_tasks: {test_loader.keys()}".replace('dict_keys', ''))

        if fwd_pass_results:
            self.logger.info(f"test_tasks_{key} (fwd_pass): {fwd_pass_results}")
            self.logger.info(f"Average test_tasks_{key} (fwd_pass): {round(sum(fwd_pass_results)/len(fwd_pass_results), 4)}")
        
        self.logger.info(f"test_tasks_{key}: {test_results}")
        self.logger.info(f"Average test_tasks_{key}: {round(sum(test_results)/len(test_results), 4)}")
    
    
    def _initialize_epi_variables(self):
        self.query_size = self.model.config.hidden_size
        self.task_infer_acc = [[] for _ in self.task_list]
        self.task_embeds = defaultdict(list)
        self.task_labels = defaultdict(list)
        
        self.task_means_over_classes = nn.ParameterList()
        # share covariance acrosss all tasks
        self.accumulate_shared_covs = nn.Parameter(torch.zeros(
            self.query_size, self.query_size), requires_grad=False)
        self.cov_inv = nn.Parameter(torch.ones(
            self.query_size, self.query_size), requires_grad=False)
     
     
    def _statistics_task_mean_cov(self, task, task_id):
        task_embeds = torch.tensor(self.task_embeds[task])
        task_labels = torch.tensor(self.task_labels[task])
        labels_space = torch.unique(task_labels)
        
        task_mean = task_embeds.mean(dim=0)
        task_cov = torch.cov((task_embeds - task_mean).T)
        
        mean_over_classes = []
        cov_over_classes = []
        for l in labels_space:
            embeds_l = task_embeds[task_labels==l]
            if embeds_l.numel() > 0 and embeds_l.shape[0] > 1:
                mean_l = embeds_l.mean(dim=0) 
                cov_l = torch.cov((embeds_l - mean_l).T)
            else:
                mean_l = task_mean
                cov_l = task_cov
                
            mean_over_classes.append(mean_l)
            cov_over_classes.append(cov_l)

        mean_over_classes = torch.stack(mean_over_classes) # N_class x dim
        shared_cov = torch.stack(cov_over_classes).mean(dim=0) # dim x dim (averaged over N_class)
        
        self.task_means_over_classes.append(nn.Parameter(mean_over_classes.cuda(), requires_grad=False))
        self.accumulate_shared_covs.data = self.accumulate_shared_covs.data.cpu()
        self.accumulate_shared_covs += shared_cov
        
        self.cov_inv = nn.Parameter(torch.linalg.pinv(
            self.accumulate_shared_covs / (task_id+1), hermitian=True), requires_grad=False) # E: Computes the pseudoinverse (Moore-Penrose inverse) 
    
    
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
        
        # prob_over_tasks = (scores_over_tasks*(-1)).T.softmax(dim=-1)
        
        return ids
    
    def _print_task_infer_acc(self):
        task_infer_acc_all = []
        for task_id, task in enumerate(self.task_list):
            avg_acc = round(sum(self.task_infer_acc[task_id])/len(self.task_infer_acc[task_id]), 4)
            task_infer_acc_all.append(avg_acc)
        self.logger.info(f"***********************************EPI task inference accuracy***********************************")
        self.logger.info(f"EPI per-task acc: {task_infer_acc_all}")
        avg_acc_all = round(sum(task_infer_acc_all)/len(task_infer_acc_all), 4)
        self.logger.info(f"Average EPI task inference acc: {avg_acc_all}")
        
    
    def _save_weight_dict(self, weight_matrix=None, valid_only=False):
        """
        Convert the list of weight tensors into a dictionary with a nested structure and save it as a JSON file.
        """
        def tensor_to_value(tensor):
            """Convert a tensor to its real value."""
            real_value = tensor.item() if isinstance(tensor, torch.Tensor) else tensor
            return round(real_value, 4)
        
        weight_list = weight_matrix if weight_matrix is not None else self.model.key_encoder.attn_weights
        # Ensure the lists are of the same length
        assert len(weight_list) == len(self.task_list)
        
        # Create the nested dictionary with tensors converted to lists
        weight_dict = {}
        for i, task in enumerate(self.task_list):
            weight_dict[task] = {self.task_list[j]: tensor_to_value(weight_list[i][j]) for j in range(len(weight_list[i])) if j<=i}
        
        output_path = os.path.join(self.args.output_dir, "weight_dict_")
        output_path += "all.json" if not valid_only else "valid_only.json"
        
        with open(output_path, 'w') as f:
            json.dump(weight_dict, f)
            
        return weight_dict


    def clean_gpu_memory(self):
        """Print and clear GPU memory usage at the end of each epoch"""
        # Assuming interest is in the first GPU, adjust or extend this function for monitoring multiple GPUs
        device_id = 0
        total_memory = torch.cuda.get_device_properties(device_id).total_memory / 1e9  # Total memory in GB
        allocated_memory = torch.cuda.memory_allocated(device_id) / 1e9  # Allocated memory
        cached_memory = torch.cuda.memory_reserved(device_id) / 1e9  # Cached memory
        print(f"GPU {device_id}: Before clearing cache: Total Memory: {total_memory:.0f} GB, Allocated: {allocated_memory:.0f} GB, Cached: {cached_memory:.0f} GB")
        
        # Clear unused cache
        torch.cuda.empty_cache()
        
        # Retrieve and print memory usage again after clearing the cache
        allocated_memory = torch.cuda.memory_allocated(device_id) / 1e9  # Update allocated memory
        cached_memory = torch.cuda.memory_reserved(device_id) / 1e9  # Update cached memory
        print(f"GPU {device_id}: After clearing cache: Total Memory: {total_memory:.0f} GB, Allocated: {allocated_memory:.0f} GB, Cached: {cached_memory:.0f} GB")