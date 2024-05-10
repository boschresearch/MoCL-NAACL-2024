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
import numpy as np
from collections import defaultdict

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
    padding_strategy,
    max_seq_length,
    max_target_length,
    pad_to_max_length,
    add_dataset_name, # TODO: explicitly give dataset name in the input examples or not
    overwrite_cache,
):
    dataset_dir = "datasets/mtl15"
    task_labels = {}
    task_labels2id = defaultdict(dict)

    for ti, task in enumerate(task_list):
        label_path = os.path.join(dataset_dir, task, 'labels.json')
        with open(label_path, 'r') as f:
            labels = json.load(f)
            task_labels[task] = labels
        task_labels2id[task] = {label:i for i, label in enumerate(task_labels[task])}

    # Padding strategy
    tokenizer.padding_side = 'left'
    label_pad_token_id = -100
    
    def get_valid_seq_len(dataset, pad_token_id):
        max_valid_len = 0
        for input_id in dataset['input_ids']:
            valid_len = len([i for i in input_id if i!=pad_token_id])
            max_valid_len = max(max_valid_len, valid_len)
        return max_valid_len

    def preprocess_function(examples):
        max_len = max_seq_length
        limit_input_len = max_seq_length + max_target_length if mode =='train' else max_seq_length
        
        texts = examples['sentence']
        labels =  examples['label']
        
        sources = []
        targets = []
        label_lens = []
        max_len = -1
        
        for text, label in zip(texts, labels):
            # Add bos and eos
            task_input = tokenizer.bos_token + text + "\nAnswer:"
            label = label + tokenizer.eos_token
            
            tokenized_input = tokenizer(task_input)["input_ids"]
            tokenized_label = tokenizer(label)["input_ids"]
            
            if mode == 'train':
                if len(tokenized_input) + len(tokenized_label) <= limit_input_len:
                    max_len = max(len(tokenized_input) + len(tokenized_label), max_len)
                    label_lens.append(len(tokenized_label))
                    sources.append(task_input + label)
                else:
                    max_len = max_seq_length
                    input_w_label = tokenizer.decode(
                        (tokenized_input + tokenized_label)[: limit_input_len],
                        skip_special_tokens=False
                    )
                    sources.append(input_w_label)
                    # if len(tokenized_input) > limited_input_len, then the source sentence will not include label
                    # Such samples will also have no loss via loss_mask
                    label_lens.append(max(0, limit_input_len - len(tokenized_input)))
            else:
                label_lens.append(0)
                if len(tokenized_input) <= limit_input_len:
                    max_len = max(len(tokenized_input), max_len)
                    sources.append(task_input)
                else:
                    max_len = limit_input_len
                    input_wo_label = tokenizer.decode(
                        tokenized_input[: limit_input_len],
                        skip_special_tokens=False
                    )
                    sources.append(input_wo_label)
                
        model_inputs = tokenizer(
            sources,
            max_length = max_seq_length,
            padding = padding_strategy,
            return_tensors = 'pt',
            truncation=True,
        )
        targets = tokenizer(
            labels,
            max_length = max_target_length,
            padding = padding_strategy,
            return_tensors = 'pt',
            truncation=True,
        )["input_ids"]

        label_mask = model_inputs["attention_mask"].bool()
        model_inputs["label"] = model_inputs['input_ids'].masked_fill(~label_mask, label_pad_token_id)

        # loss mask
        # max_len = min(max_len, limit_input_len)
        max_len = min(label_mask.shape[-1], limit_input_len)
        loss_mask = torch.ones((label_mask.shape))
        for k, label_len in enumerate(label_lens):
            loss_mask[k, : max_len - label_len - 1] = 0
        model_inputs['loss_mask'] = loss_mask.masked_fill(~label_mask, 0)
        model_inputs['targets'] = targets
        return model_inputs
        

    def get_dataset_target_ids(dataset: list, task):
        for i, sample in enumerate(dataset):
            if 'label' not in sample or sample['label'] not in task_labels2id[task].keys():
                dataset.pop(i)
                i -= 1
            else:
                sample['target_ids'] = task_labels2id[task][sample['label']]
                dataset[i] = sample
            
        return dataset
    
    def select_subset_dataset(dataset, task, n_per_class, seed):
        np.random.seed(seed)
        
        labels = task_labels[task]
        dataset_labels = [item['label'] for item in dataset]
        train_idx = []
        
        for label in labels:
            idx_total = [i for i, l in enumerate(dataset_labels) if l == label]
            np.random.shuffle(idx_total)

            train_pool = idx_total
            if n_per_class < 0:
                train_idx.extend(train_pool)
            else:
                train_idx.extend(train_pool[:n_per_class])
        np.random.shuffle(train_idx)
        
        dataset = [dataset[idx] for idx in train_idx]
        dataset = get_dataset_target_ids(dataset, task)
        
        return dataset
    
    def get_dataloader(dataset, mode):
        dataset = Dataset.from_list(dataset)
        dataset = dataset.map(
            lambda x: preprocess_function(x),
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

    data_collator = default_data_collator if pad_to_max_length else DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    dataloader = {}
    valid_seq_len = {}
    
    n_train_per_class = 16 if data_args.n_train_per_class is None else data_args.n_train_per_class
    n_val_per_class = 50 if data_args.n_val_per_class is None else data_args.n_val_per_class

    
    for ti, task in enumerate(task_list):
        dataloader[task] = {}
        
        
        for mode in mode_list:
            print(f"processing [{task}-{mode}] dataset...")
            
            dataset_path = os.path.join(dataset_dir, task, mode+'.json')
            with open(dataset_path, 'r') as f:
                dataset = json.load(f)
                
            if mode == 'train':
                train_dataset = select_subset_dataset(dataset, task, n_train_per_class, training_args.seed)
                dataloader[task]['train'] = get_dataloader(train_dataset, mode='train')
                
                valid_seq_len[task] = get_valid_seq_len(dataloader[task]['train'].dataset, tokenizer.pad_token_id)
            
            elif mode == 'dev':
                dev_dataset = select_subset_dataset(dataset, task, n_val_per_class, training_args.seed)
                dataloader[task]['dev'] = get_dataloader(dev_dataset, mode='dev')
            
            elif mode == 'test':
                print(f"processing [{task}-test] dataset...")
                
                if data_args.n_val_per_class and data_args.n_val_per_class <= 16: # debug mode
                    dataset = dataset[:data_args.n_val_per_class]
                dataset = get_dataset_target_ids(dataset, task)
                dataloader[task]['test'] = get_dataloader(dataset, mode='test')
                
    for task, valid_len in valid_seq_len.items():
        print(f"Task [{task}]: max. valid input sequence length {valid_len}")
        
    return dataloader