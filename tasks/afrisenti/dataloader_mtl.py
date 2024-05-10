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
    multilingual_taskuage_list = None,
):
    task2id = {v: i for i, v in enumerate(task_list)}
    if type(label_list[0]) == list:
        # get_label_ids_and_offset(task_list, label_list, )
        label2id = {}
        offset = {}
        for ti, task in enumerate(task_list):
            label2id[task] =  {v: i for i, v in enumerate(label_list[ti])}
            offset[task] = len(label_list[ti-1]) + offset[task_list[ti-1]] if ti>0 else 0
    else:
        label2id = {v: i for i, v in enumerate(label_list)}
    # Padding strategy
    padding = "max_length" if pad_to_max_length else False
    

    
    def preprocess_function(examples):
        # Tokenize the texts TODO: rename the keys, store in a dict e.g, task2keys
        try:
            texts =(examples["text"],)
        except:
            texts =(examples["tweet"],)
        
        result =  tokenizer(*texts, padding=padding, max_length=max_seq_length, truncation=True)
        # Map labels to IDs
        if label2id is not None and "label" in examples:
            if type(label_list[0]) == list:
                result["label"] = [(label2id[task][l] if l != -1 else -1) for l in examples["label"]]
            else:
                result["label"] = [(label2id[l] if l != -1 else -1) for l in examples["label"]]
        return result

    # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
    if pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    dataloader = {}
    
    for ti, task in enumerate(task_list):
        dataloader[task] = {}
        for mode in mode_list:
            dataset_path = os.path.join("datasets/afrisenti", f"{mode}/{task}_{mode}.tsv")
            dataset = pd.read_csv(dataset_path, sep='\t').dropna()
            
            if mode=='train' and data_args.max_train_samples is not None:
                max_samples = min(len(dataset), data_args.max_train_samples)
                dataset = dataset.sample(n=max_samples)
            elif mode=='dev' and data_args.max_eval_samples is not None:
                max_samples = min(len(dataset), data_args.max_eval_samples)
                dataset = dataset.sample(n=max_samples)
            
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
            dataloader[task][mode] = DataLoader(dataset, 
                                                batch_size=batch_size,
                                                sampler=sampler, 
                                                collate_fn=data_collator,
                                                drop_last=drop_last,
                                                num_workers=training_args.dataloader_num_workers,
                                                pin_memory=training_args.dataloader_pin_memory,
                                                worker_init_fn=worker_init_fn,
                                                )
    return dataloader