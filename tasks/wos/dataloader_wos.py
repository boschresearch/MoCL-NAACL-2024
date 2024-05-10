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
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler, SequentialSampler

from datasets import Dataset
from transformers import (
    DataCollatorWithPadding,
    default_data_collator,
)
from transformers.trainer_utils import set_seed

def seed_worker(_):
    """
    Helper function to set worker seed during Dataloader initialization.
    """
    worker_seed = torch.initial_seed() % 2**32
    set_seed(worker_seed)


def DataLoaderMTL(
    data_args,
    training_args,
    task_list,
    label_list,
    mode_list,
    tokenizer,
    max_seq_length,
    pad_to_max_length,
    overwrite_cache,
    logger = None
):
    assert type(label_list[0]) == list
    label2id = {}
    offset = {}
    for ti, task in enumerate(task_list):
        label2id[task] =  {v: i for i, v in enumerate(label_list[ti])}
        offset[task] = len(label_list[ti-1]) + offset[task_list[ti-1]] if ti>0 else 0

    # Padding strategy
    padding = "max_length" if pad_to_max_length else False
    
    def preprocess_function(examples):
        examples['text'] = examples['input_data']
        
        result = tokenizer(examples['text'], 
                            padding=padding, 
                            max_length=max_seq_length, 
                            truncation=True)
            
        # Map labels to true IDs
        if label2id is not None and "label_level_2" in examples:
            result["labels"] = [int(l) for l in examples['label_level_2']]
        result["label"] = [int(l) for l in examples['label_level_2']]
        return result
    
    
    def get_dataloader(dataset, mode):
        dataset = Dataset.from_pandas(dataset)
        dataset = dataset.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=not overwrite_cache,
            desc="Running tokenizer on train dataset",
            )
        
        sampler = RandomSampler(dataset) if mode in ['train'] else SequentialSampler(dataset)
        drop_last = training_args.dataloader_drop_last if mode in ['train', 'dev'] else False
        worker_init_fn = seed_worker if mode == 'train' else None
        batch_size = training_args.per_device_train_batch_size if mode == 'train' else training_args.per_device_eval_batch_size
        
        return DataLoader(dataset,
                          batch_size=batch_size,
                          sampler=sampler, 
                          collate_fn=data_collator,
                          drop_last=drop_last,
                          num_workers=training_args.dataloader_num_workers,
                          pin_memory=training_args.dataloader_pin_memory,
                          worker_init_fn=worker_init_fn,
                        )


    # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
    if pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    dataset = {}
    dataloader = {}
    
    dataset_path = os.path.join("datasets/wos/WOS11967", "wos.csv")
    dataset_all = pd.read_csv(dataset_path, header=None, names=['input_data', 'label', 'label_level_1', 'label_level_2'])
    
    for ti, task in enumerate(task_list):
        dataloader[task] = {}
        task_idx = np.where(np.array(dataset_all["label_level_1"]==task))
        dataset[task] = dataset_all.iloc[task_idx]
        num_examples = len(dataset[task])
        
        n_train, n_val, n_test = int(num_examples*0.6), int(num_examples*0.2), int(num_examples*0.2)
        
        test_dataset = dataset[task].iloc[-n_test:]
        np.random.seed(training_args.seed)
        train_val_idx = np.array(range(n_train+n_val))
        np.random.shuffle(train_val_idx)
        train_idx, val_idx = train_val_idx[:n_train], train_val_idx[n_train:]
        
        train_dataset = dataset[task].iloc[train_idx]
        val_dataset = dataset[task].iloc[val_idx]
        
        print(f"processing task-[{task}] dataset...")
        print(f"Train dataset size: {n_train}")
        print(f"Validation dataset size: {n_val}")
        print(f"Test dataset size: {n_test}")
        
        if logger is not None:
            logger.info(f"Task-[{task}]-train first 2 examples: \n {train_dataset[:2]['input_data'].values}")
            logger.info(f"Task-[{task}]-test first 2 examples: \n {test_dataset[:2]['input_data'].values}")
                
        
        dataloader[task]['train'] = get_dataloader(train_dataset, mode='train')
        dataloader[task]['dev'] = get_dataloader(val_dataset, mode='dev')
        dataloader[task]['test'] = get_dataloader(test_dataset, mode='test')

    return dataloader