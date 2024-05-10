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
import sys
sys.path.append(".")
sys.path.append("../")

from arguments import get_args
from utils.set_logger import set_logger
from transformers import AutoConfig, AutoTokenizer, set_seed

from model.utils import get_model, TaskType
from training.trainer_continual_full_ft_bert_new import ContinualTrainerMTL
from tasks.wos.dataloader_wos import DataLoaderMTL


if __name__ == "__main__":
    args = get_args()
    model_args, data_args, training_args = args
    
    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir, exist_ok=True)
    
    logfile = os.path.join(training_args.output_dir, "log.txt")
    logger = set_logger(logfile)
        
    config_path = os.path.join(training_args.output_dir, f"configs.json")
    with open(config_path, "a", newline='\n') as f:
        f.write(f"\nmodel_args:\n {model_args}\n")
        f.write(f"\ndata_args:\n {data_args}\n")
        f.write(f"\ntraining_args:\n {training_args}\n")


    task_list = model_args.mtl_task_list.split('_')
    model_args.cl_language_list = model_args.mtl_task_list # TODO: make them consistent and delete this
    mode_list = ["train", "test"]
    task2labels = {f'{n}': list(range(5)) for n in list(range(7))}
    label_list = [[] for _ in task_list]
    num_labels = 0
    for ti, task in enumerate(task_list):
        label_list[ti] = task2labels[task]
        num_labels += len(label_list[ti])
    
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    
    # Set seed before initializing model.
    seed = training_args.seed
    set_seed(seed)
    
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        num_labels=num_labels,
        revision=model_args.model_revision,
    )
    
    model_args.prompt_save_path = os.path.join(os.path.dirname(training_args.output_dir), "prompts")
    if not os.path.exists(model_args.prompt_save_path):
        os.mkdir(model_args.prompt_save_path)

    model = get_model(
        model_args, 
        task_type=TaskType.SEQUENCE_CLASSIFICATION, 
        config=config, 
        seed=seed,
        mtl=True, 
        mtl5=True,
        max_seq_len=data_args.max_seq_length,
        output_dir=training_args.output_dir,
        label_list=label_list
        )
    
    dataloaders = DataLoaderMTL(
        data_args,
        training_args,
        task_list,
        label_list,
        mode_list,
        tokenizer,
        data_args.max_seq_length,
        data_args.pad_to_max_length,
        data_args.overwrite_cache,
        logger=logger
    )
    
    train_dataloaders = {task: dataloaders[task]['train'] for task in task_list}
    dev_dataloaders = {task: dataloaders[task]['dev'] for task in task_list}
    test_dataloaders = {task: dataloaders[task]['test'] for task in task_list}
    
    trainer = ContinualTrainerMTL(
        training_args,
        model,
        logger,
        task_list,
        label_list,
        data_args.early_stopping_patience if data_args.early_stop else -1,
        data_args.learning_rate_list
    )
    trainer.train(
        train_dataloaders,
        dev_dataloaders,
        test_dataloaders,
    )