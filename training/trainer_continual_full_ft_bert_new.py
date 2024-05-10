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
import math
import shutil
import numpy as np

import torch
import torch.nn as nn

from datetime import datetime
from tqdm import tqdm

from transformers import EvalPrediction
from transformers.trainer_pt_utils import get_parameter_names
from training.early_stopping import EarlyStopping


class ContinualTrainerMTL(nn.Module):
    def __init__(self,
                 args,
                 model,
                 logger,
                 task_list,
                 label_list, 
                 early_stopping_patience=-1,
                 learning_rate_list=None):
        super(ContinualTrainerMTL, self).__init__()
        
        self.args = args
        self.seed = args.seed
        self.model = model.to(self.args.device)
        self.logger = logger
        self.task_list = task_list
        self.num_tasks = len(task_list)
        
        self.num_train_epochs = math.ceil(args.num_train_epochs)
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping = EarlyStopping(
            save_path=os.path.join(args.output_dir, 'best_checkpoint'),
            logger=self.logger,
            patience = early_stopping_patience
        )
        try:
            self.learning_rate_list = [float(x) for x in learning_rate_list.split('_')]
        except:
            self.learning_rate_list = [self.args.learning_rate for _ in range(self.num_tasks)]
        
        self.logger.info(f"***********************************seed: {self.seed}***********************************")

    
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
        
        # if self.args.optim == OptimizerNames.ADAMW_HF:
        from torch.optim import AdamW

        optimizer_cls = AdamW
        optimizer_kwargs.update(adam_kwargs)

        # Initilaize optimizer
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)


    def _process_data(self, loader, task_id=0):
        try:
            data_batch = next(loader[1])
        except:
            loader[1] = iter(loader[0])
            data_batch = next(loader[1])
        try:
            inputs = {"input_ids": data_batch["input_ids"].to(self.args.device), 
                        "attention_mask": data_batch["attention_mask"].to(self.args.device), 
                        "token_type_ids": data_batch["token_type_ids"].to(self.args.device),
                        "labels": data_batch["labels"].to(self.args.device)}
        except:
            inputs = {"input_ids": data_batch["input_ids"].to(self.args.device), 
                        "attention_mask": data_batch["attention_mask"].to(self.args.device), 
                        "labels": data_batch["labels"].to(self.args.device)}
        
        label = data_batch["labels"].to(self.args.device)
        
        return inputs, label

    
    def compute_loss(self, model, inputs, task_id, mode, final=False):
        inputs['return_dict'] = True
        
        outputs = model(**inputs)
            
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        
        return loss, outputs
    
    
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
        
        
        fwd_pass_results_acc = [0. for _ in range(self.num_tasks)]
        if self.args.do_train:
            for task_id, task in enumerate(self.task_list):
                num_train_batch = train_batch[task_id]
                num_training_steps = self.num_train_epochs * num_train_batch
                num_examples_dataset = train_batch[task_id] * self.args.per_device_train_batch_size
                num_examples_training = num_training_steps * self.args.per_device_train_batch_size
            
                self._prepare_optimizer(task_id)

                self.model.train_loss_buffer = np.zeros([self.num_tasks, self.num_train_epochs]) # needed?
                self.model.epochs = self.num_train_epochs
                self.model.task_id = task_id
                self.early_stopping.reinit()
                eval_acc, test_acc = [], []

                # Train!
                self.logger.info(f"***** Running training - task {task_id}: {task}*****")
                self.logger.info(f"  Num Epochs = {self.num_train_epochs}")
                self.logger.info(f"  Num examples datasets = {num_examples_dataset}")
                self.logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size}")
                self.logger.info(f"  Total optimization steps = {num_training_steps}")
                self.logger.info(f"  Num examples training = {num_examples_training}")

                
                for epoch in range(self.num_train_epochs):
                    self.model.epoch = epoch
                    self.model.train()
                    train_preds, train_gts = [], []
                    
                    for batch_index in tqdm(range(num_train_batch), total=num_training_steps):
                        train_input, train_gt = self._process_data(train_loader[task], task_id)
                        
                        train_loss, train_pred = self.compute_loss(self.model, train_input, task_id, mode='train')

                        train_preds.append(train_pred)
                        train_gts.append(train_gt)
                        
                        self.model.zero_grad()
                        train_loss.backward()
                        self.optimizer.step()
                    preds = np.concatenate(np.array([pred.logits.view(-1, pred.logits.shape[-1]).detach().to("cpu").numpy() for pred in train_preds], dtype=object), axis=0)
                    label_ids = np.concatenate(np.array([label_id.view(-1).detach().to("cpu").numpy() for label_id in train_gts], dtype=object), axis=0)
                    
                    metrics = self.compute_metrics(EvalPrediction(predictions=preds, label_ids=label_ids))
                    # self._display_results(metrics, task_id, mode='train')
                    
                    eval_acc_ = self.eval(val_loader[task], val_batch[task_id], task_id, mode='val')
                    eval_acc.append(eval_acc_)

                    best_epoch = epoch
                    if self.early_stopping_patience > 0 and (self.early_stopping(eval_acc_, self.model, task) or epoch == self.num_train_epochs-1):
                        best_acc_eval = self.early_stopping.best_score
                        best_epoch = eval_acc.index(best_acc_eval)
                        
                        self.logger.info(f"Early stop!!! Best epoch for [{task}]: {best_epoch}")
                        self.logger.info(f"Best evaluation accuracy [{task}]: {round(best_acc_eval, 4)}")
                    
                        # load weights of the best model
                        model_path = os.path.join(self.early_stopping.save_path, f'best_checkpoint_{task}.pth')
                        self.logger.info(f"Loading weights from the best model (best epoch {best_epoch})...")
                        self.model.load_state_dict(torch.load(model_path))
                        if task_id != len(self.task_list) - 1:
                            os.remove(model_path)
                            
                        # evaluate on the test set using the best model; 
                        best_acc_test = self.eval(test_loader[task], test_batch[task_id], task_id, mode='test')
                        
                        # self.logger.info(f"Best evaluation accuracy [{task}]: {round(best_acc_eval, 4)}")
                        self.logger.info(f"Best test accuracy [{task}]: {round(best_acc_test, 4)}")
                        fwd_pass_results_acc[task_id] = round(best_acc_test, 4)
                        
                        
                        break

        
        test_results_acc = []
        for t_id, t in enumerate(test_loader.keys()):
            
            self.logger.info(f"Final evaluation (seed-{self.seed}) on task [{t}]")
            test_acc = self.eval(test_loader[t], test_batch[t_id], t_id, mode='test', final=True)
            test_results_acc.append(round(test_acc, 4))
        
        self._log_final_results(test_loader, test_results_acc, fwd_pass_results_acc, key='acc')
        
        # remove checkpoints
        shutil.rmtree(self.early_stopping.save_path)
    
    
    def eval(self, loader, batch, task_id, mode='test', final=False):
        self.model.eval()
        preds, gts = [], []
        with torch.no_grad():
            for batch_index in tqdm(range(batch)):
                input, gt = self._process_data(loader)
                loss, pred = self.compute_loss(self.model, input, task_id, mode=mode, final=final)
                
                preds.append(pred)
                gts.append(gt)
        preds = np.concatenate(np.array([pred.logits.view(-1, pred.logits.shape[-1]).detach().to("cpu").numpy() for pred in preds], dtype=object), axis=0)
        label_ids = np.concatenate(np.array([label_id.view(-1).detach().to("cpu").numpy() for label_id in gts], dtype=object), axis=0)
            
        metrics = self.compute_metrics(EvalPrediction(predictions=preds, label_ids=label_ids))
        self._display_results(metrics, task_id, mode, final)
        
        return metrics["accuracy"]

    
    def compute_metrics(self, p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)
        
        now = datetime.now()
        accuracy = (preds == p.label_ids).astype(np.float32).mean().item()

        return {
            "accuracy": accuracy,
        }
    
    
    def _display_results(self, metrics, task_id, mode, final=False):
        if final:
            self.logger.info(f"***** Task {self.task_list[task_id]} - Final evaluation: {mode} *****")
        else:
            self.logger.info(f"***** Task {self.task_list[task_id]} - Epoch {self.model.epoch}: {mode} (seed {self.seed}) *****")
        
        accuracy = metrics["accuracy"]
        
        self.logger.info(f"Accuracy: {round(accuracy, 4)}")
    
    
    def _log_final_results(self, test_loader, test_results, fwd_pass_results=None, key='acc', task_identify_results=None):
        self.logger.info(f"***********************************seed: {self.seed} - {key}***********************************")
        
        self.logger.info(f"Test_tasks: {test_loader.keys()}".replace('dict_keys', ''))
        
        if fwd_pass_results:
            self.logger.info(f"test_tasks_{key} (fwd_pass): {fwd_pass_results}")
            self.logger.info(f"Average test_tasks_{key} (fwd_pass): {round(sum(fwd_pass_results)/len(fwd_pass_results), 4)}")

        self.logger.info(f"test_tasks_{key}: {test_results}")
        self.logger.info(f"Average test_tasks_{key}: {round(sum(test_results)/len(test_results), 4)}")