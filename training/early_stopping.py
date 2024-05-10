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
import torch
from copy import deepcopy     
from mpeft.utils.save_and_load import get_peft_model_state_dict

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, save_path, logger, patience=5, verbose=False, delta=0):
        """
        Args:
            save_path : path to the model save directory
            patience (int): How long to wait after last time validation loss improved.
                            Default: 5
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.save_path = save_path
        self.logger = logger
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_model = None
        self.early_stop = False
        self.val_score_max = 0.
        self.delta = delta
        
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)


    def __call__(self, score, model, task, save=True):

        if self.best_score is None:
            self.best_score = score
            if save:
                self.save_checkpoint(score, model, task)
            
        elif score <= self.best_score + self.delta:
            self.counter += 1
            self.logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            if save:
                self.save_checkpoint(score, model, task)
            self.counter = 0
        
        return self.early_stop


    def save_checkpoint(self, score, model, task):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.logger.info(f'Validation score increased ({self.val_score_max:.4f} --> {score:.4f}).  Saving model ...')

        path = os.path.join(self.save_path, f'best_checkpoint_{task}.pth')
        try:
            lora_params = get_peft_model_state_dict(model, adapter_name=task)
            # torch.save(lora_params, path)
            self.best_model = deepcopy(lora_params)
        except:
            torch.save(model.state_dict(), path)
        
        self.val_score_max = score


    def reinit(self):
        self.counter = 0
        self.best_score = None
        self.best_model = None
        self.early_stop = False
        self.val_score_max = 0.
        