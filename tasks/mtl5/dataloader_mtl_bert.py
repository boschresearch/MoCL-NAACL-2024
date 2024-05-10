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
    task_keys = {
        'yelp': ['labels', 'content'],
        'amazon': ['labels', 'title', 'content'],
        'yahoo': ['labels', 'title', 'content', 'answer'],
        'agnews': ['labels', 'title', 'content'],
        'dbpedia': ['labels', 'title', 'content']
    }
    task_labels = {
        'yelp': ["terrible", "bad", "middle", "good", "wonderful"],
        'amazon': ["terrible", "bad", "middle", "good", "wonderful"],
        'yahoo': ["society and culture", "science", "health", "education and reference", 
                  "computers and internet", "sports", "business", "entertainment and music", 
                  "family and relationships", "politics and government"],
        'agnews': ["world", "sports", "business", "science"],
        'dbpedia': ["company", "educationalinstitution", "artist", "athlete", "officeholder", 
                    "meanoftransportation", "building", "naturalplace", "village", "animal",
                    "plant", "album", "film", "writtenwork"]
    }
    task2id = {v: i for i, v in enumerate(task_list)}

    assert type(label_list[0]) == list
    # get_label_ids_and_offset(task_list, label_list, )
    label2id = {}
    offset = {}
    for ti, task in enumerate(task_list):
        label2id[task] =  {v: i for i, v in enumerate(label_list[ti])}
        offset[task] = len(label_list[ti-1]) + offset[task_list[ti-1]] if ti>0 else 0

    # Padding strategy
    padding = "max_length" if pad_to_max_length else False
    

    
    def preprocess_function(examples):
        if task == 'yahoo':
            examples['text'] = []
            for title, content, answer in zip(examples['title'], examples['content'], examples['answer']):
                examples['text'].append(f'{title}[SEP]{content}[SEP]{answer}')

        elif len(task_keys[task]) == 3:
            examples['text'] = []
            for title, content in zip(examples['title'], examples['content']):
                examples['text'].append(f'{title}[SEP]{content}')
        else:
            examples['text'] = examples['content']
        
        text = examples["content"]
        result = tokenizer(text, 
                            padding=padding, 
                            max_length=max_seq_length, 
                            truncation=True)
            
        # Map labels to true IDs
        if label2id is not None and "labels" in examples:
            result["labels"] = [l - 1 for l in examples['labels']]
        return result
    
    def select_subset_dataset_per_label(dataset, task, n_train_per_class, n_val_per_class, seed):
        np.random.seed(seed)
        
        n_labels = len(task_labels[task])
        train_idx, val_idx = [], []
        
        for l in range(n_labels):
            idx_total = np.where(np.array(dataset["labels"])==(l+1))[0]
            np.random.shuffle(idx_total)

            train_pool = idx_total[:-n_val_per_class]
            if n_train_per_class < 0:
                train_idx.extend(train_pool)
            else:
                train_idx.extend(train_pool[:n_train_per_class])
            val_idx.extend(idx_total[-n_val_per_class:])
        
        np.random.shuffle(train_idx)
        np.random.shuffle(val_idx)
        
        return train_idx, val_idx
    
    
    def select_subset_dataset(dataset, task, n_train_total, n_val_per_class, seed):
        np.random.seed(seed)
        
        n_labels = len(task_labels[task])
        n_train_per_class = n_train_total // n_labels
        
        train_idx, val_idx = [], []
        
        for l in range(n_labels):
            idx_total = np.where(np.array(dataset["labels"])==(l+1))[0]
            np.random.shuffle(idx_total)

            train_pool = idx_total[:-n_val_per_class]
            if n_train_per_class < 0:
                train_idx.extend(train_pool)
            else:
                if l == n_labels-1:
                    train_idx.extend(train_pool[:n_train_total - n_train_per_class*(n_labels-1)])
                else:
                    train_idx.extend(train_pool[:n_train_per_class])
            val_idx.extend(idx_total[-n_val_per_class:])
        
        np.random.shuffle(train_idx)
        np.random.shuffle(val_idx)
        
        return train_idx, val_idx
        
    
    
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

    dataloader = {}
    
    for ti, task in enumerate(task_list):
        dataloader[task] = {}
        for mode in mode_list:
            dataset_path = os.path.join("datasets/mtl5", f"{mode}/{task}.csv").replace('dev', 'train')
            dataset = pd.read_csv(dataset_path, header=None, names=task_keys[task])

            dataset.dropna(subset=['content'], inplace=True)
            
            # filter rows with length greater than 20 (2 words including spaces on average)
            dataset.drop(dataset[dataset['content'].map(len) < 20].index, inplace=True)

            if mode == 'train':
                # Set 'max_samples' limit, train=115000, dev=7600 for each task
                n_train_per_class = 2000 if data_args.n_train_per_class is None else data_args.n_train_per_class
                n_val_per_class = 500 if data_args.n_val_per_class is None else data_args.n_val_per_class
                if data_args.n_train_total is not None:
                    train_idx, val_idx = select_subset_dataset(dataset, task, data_args.n_train_total, n_val_per_class, training_args.seed)
                else:
                    train_idx, val_idx = select_subset_dataset_per_label(dataset, task, n_train_per_class, n_val_per_class, training_args.seed)
                train_dataset = dataset.iloc[train_idx]
                dev_dataset = dataset.iloc[val_idx]
                
                print(f"processing [{task}-train-dev] dataset...")
                print(f"Train dataset size: {len(train_idx)}")
                print(f"Validation dataset size: {len(val_idx)}")
                if logger is not None:
                    logger.info(f"Task [{task}]-train first 2 examples: \n {train_dataset[:2]['content'].values}")
                
                dataloader[task]['train'] = get_dataloader(train_dataset, mode='train')
                dataloader[task]['dev'] = get_dataloader(dev_dataset, mode='dev')
            elif mode == 'test':
                print(f"processing [{task}-test] dataset...")
                if data_args.n_val_per_class and data_args.n_val_per_class < 50: # debug mode
                    dataset = dataset.iloc[:data_args.n_val_per_class]
                if data_args.max_predict_samples is not None:
                    dataset = dataset.sample(frac=1).iloc[:data_args.max_predict_samples]
                dataloader[task]['test'] = get_dataloader(dataset, mode='test')
                
    return dataloader