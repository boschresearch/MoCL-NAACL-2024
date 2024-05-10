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

import json
from transformers import LlamaTokenizer, HfArgumentParser, TrainingArguments, set_seed
from mpeft import LoraConfig, KeyEncoderConfig, TaskType
from arguments import DataTrainingArguments, ModelArguments

from utils.set_logger import set_logger
from model.modeling_llama import LlamaModel
from model.causal_lm_llama import LlamaContinualForCausalLM
from training.trainer_continual_causal_llama_lora import ContinualTrainerMTL
from tasks.mtl5.dataloader_mtl_causal_llama import DataLoaderMTL


if __name__ == "__main__":
    
    ### 1. Load all required configs ###
    parser = HfArgumentParser((TrainingArguments, DataTrainingArguments, ModelArguments))
    training_args, data_args, model_args = parser.parse_args_into_dataclasses()
    
    # Set seed
    seed = training_args.seed
    set_seed(seed)
    
    # Create output directories
    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir, exist_ok=True)
    lora_output_dir = os.path.join(training_args.output_dir, 'lora')
    if not os.path.exists(lora_output_dir):
        os.mkdir(lora_output_dir)
    
    # Load peft and mpeft configs
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=4, lora_alpha=16, lora_dropout=0.1, bias="none", 
        output_dir=lora_output_dir,
        target_modules=[
            "q_proj",
            "v_proj",  
        ],
        # for mixture of peft modules
        mpeft_enabled=model_args.mpeft_enabled,
        )
    if model_args.mpeft_enabled:
        mpeft_config = KeyEncoderConfig(
            seed=seed,
            query_encoder_type=model_args.query_encoder_type,
            # matching_loss_v2=model_args.matching_loss_v2,
            task_list=data_args.task_list.split('_')
            )
        # Merge peft_config and mpeft_config
        peft_config.mpeft_config = mpeft_config
        
    
    logfile = os.path.join(training_args.output_dir, "log.txt")
    logger = set_logger(logfile)
        
    config_path = os.path.join(training_args.output_dir, f"configs.json")
    # Save the configurations
    with open(config_path, "a", newline='\n') as f:
        f.write(f"\n(m)peft_args:\n {peft_config}\n")
        f.write(f"\ntraining_args:\n {training_args}\n")
 
 
    ### 2. Load dataset ###
    task_list = data_args.task_list.split('_')
    mode_list = ["train", "dev", "test"]
    dataset_dir = "datasets/mtl15"
    task_labels = {}
    label_list = [[] for _ in task_list]

    for ti, task in enumerate(task_list):
        label_path = os.path.join(dataset_dir, task, 'labels.json')
        with open(label_path, 'r') as f:
            labels = json.load(f)
        task_labels[task] = labels
        label_list[ti] = labels
        
        if ti == 0:
            num_labels = len(labels)
    
    # Load Llama 2 Tokenizer
    model_name_or_path = model_args.model_name_or_path
    tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path, add_prefix_space=True)
    tokenizer.bos_token_id = 1
    tokenizer.eos_token_id = 2
    tokenizer.pad_token_id = 1
    
    if data_args.max_seq_length_list is not None:
        max_seq_length = [int(sql) for sql in data_args.max_seq_length_list.split('_')]
    else:
        max_seq_length = data_args.max_seq_length
    max_target_length = data_args.max_target_length

    dataloaders = DataLoaderMTL(
        data_args,
        training_args,
        task_list,
        label_list,
        mode_list,
        tokenizer,
        data_args.padding_strategy,
        max_seq_length,
        max_target_length,
        data_args.pad_to_max_length,
        data_args.add_dataset_name,
        data_args.overwrite_cache,
    )
    train_dataloaders = {task: dataloaders[task]['train'] for task in task_list}
    dev_dataloaders = {task: dataloaders[task]['dev'] for task in task_list}
    test_dataloaders = {task: dataloaders[task]['test'] for task in task_list}
    
    
    ### 3. Load the base model ### 
    
    model = LlamaContinualForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_name_or_path,
        device_map="auto",
        offload_folder="offload",
        trust_remote_code=True
        )
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    
    for arg in dir(model_args):
        if not arg.startswith("__") and not callable(getattr(model_args, arg)):
            setattr(model.config, arg, getattr(model_args, arg))

    # Initialize the query encoder
    query_encoder = LlamaModel(model.config)
    for param in query_encoder.parameters():
        param.requires_grad = False
        
    ### 4. Set up the trainer ###
    trainer = ContinualTrainerMTL(
        training_args,
        model,
        query_encoder,
        logger,
        task_list,
        label_list,
        peft_config,
        lora_save_dir=os.path.join(training_args.output_dir, 'checkpoint_loras'),
        early_stopping_patience=data_args.early_stopping_patience if data_args.early_stop else -1,
        tokenizer=tokenizer,
        max_target_length = data_args.max_target_length,
        learning_rate_list=data_args.learning_rate_list,
    )
    
    ### 5. Start training ###
    trainer.train(
        train_dataloaders,
        dev_dataloaders,
        test_dataloaders,
    )