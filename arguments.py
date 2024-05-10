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

from dataclasses import dataclass, field
from typing import Optional

from transformers import HfArgumentParser, TrainingArguments


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.training_args
    """
    task_list: str = field(
        default='agnews_yelp_amazon_yahoo_dbpedia',
        metadata={
            "help": "Task list in the continual learning mode, order matters (BERT order-0 by default)"
        }
    )
    dataset_path: str = field(
        default="datasets/afrisenti",
        metadata={
            "help": "Path to the dataset folder"
        }
    )
    padding_strategy: str = field(
        default="max_length", 
        metadata={"help": "Padding strategy. Choices: ['max_length', 'longest', 'do_not_pad']"}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: int = field(
        default=20,
        metadata={
            "help": "The maximum total label sequence length after tokenization (for generative models). Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    add_dataset_name: bool = field(
        default=False, metadata={"help": "Add a prefix with dataset name in the task input or not."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    n_train_total: Optional[int] = field(
        default=None,
        metadata={
            "help": "Select a subset of the dataset consisting of N samples for each task (including all classes) for training (used for MTL5 dataset, bert-based: 115000)"
        },
    )
    n_train_per_class: Optional[int] = field(
        default=None,
        metadata={
            "help": "Select a subset of the dataset consisting of N samples for each class for training (used for MTL5 dataset, bert-based: 2000, t5-based: 16)"
        },
    )
    n_val_per_class: Optional[int] = field(
        default=None,
        metadata={
            "help": "Select a subset of the dataset consisting of N samples for each class for validation (used for MTL5 dataset, bert-based: 2000, t5-based: 16)"
        },
    )
    early_stop: bool = field(
        default=True,
        metadata={
            "help": "Use early stop or not"
        }
    )
    early_stopping_patience: Optional[int] = field(
        default=5,
        metadata={
            "help": "Stop training when the specified metric worsens for [early_stopping_patience] evaluation calls"
        }
    )
    learning_rate_list: Optional[str] = field(
        default=None,
        metadata={
            "help": "Use different learning rate for different tasks"
        }
    )
    max_seq_length_list: Optional[str] = field(
        default=None,
        metadata={
            "help": "Use different max_seq_length for different tasks, to avoid catastrophic forgetting"
        }
    )

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        default="/fs/scratch/rb_bd_dlp_rng-dl01_cr_AIM_employees/model_cache/afro-xlmr-large",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    prefix: bool = field(
        default=True,
        metadata={
            "help": "Will use P-tuning v2 during training"
        }
    )
    prompt: bool = field(
        default=False,
        metadata={
            "help": "Will use prompt tuning during training"
        }
    )
    multi_peft_modules: bool = field(
        default=True,
        metadata={
            "help": "Multi-task learning with multiple prefix (for composition / concatenation / identification)"
        }
    )
    disentangle_modules: bool = field(
        default=False,
        metadata={
            "help": "Used in Llama experiments for per task fine-tuning and EPI inference. Multi-task learning with separate (lora/prefix) modules."
        }
    )
    pre_seq_len: int = field(
        default=8,
        metadata={
            "help": "The length of prompt"
        }
    )
    pre_seq_len_g: int = field(
        default=4,
        metadata={
            "help": "The length of general prompt, only works when add_general_prompt = True"
        }
    )
    prefix_projection: bool = field(
        default=True,
        metadata={
            "help": "Apply a two-layer MLP head over the prefix embeddings"
        }
    ) 
    prompt_projection: bool = field(
        default=True,
        metadata={
            "help": "Apply a two-layer MLP head over the prompt embeddings"
        }
    ) 
    residual_prefix_projection: bool = field(
        default=False,
        metadata={
            "help": "Apply a skip connection (residual) over the prefix embeddings"
        }
    ) 
    residual_prompt_projection: bool = field(
        default=False,
        metadata={
            "help": "Apply a skip connection (residual) over the prompt embeddings"
        }
    ) 
    continual_learning: bool = field(
        default=False,
        metadata={
            "help": "Run P-tuning v2 in the continual learning mode"
        }
    )
    compose_prompts: bool = field(
        default=False,
        metadata={
            "help": "Use compositional prompts (via query key match) or not"
        }
    )
    log_weights: bool = field(
        default=True,
        metadata={
            "help": "Log weights with tensorboard during training"
        }
    )
    add_general_prompt: bool = field(
        default=False,
        metadata={
            "help": "Add general prompt (in contrast to task-specific prompts) or not"
        }
    )
    add_attention_filter: bool = field(
        default=False,
        metadata={
            "help": "Add attention filters to queries or not"
        }
    )
    composed_prompt_for_usage: bool = field(
        default=False,
        metadata={
            "help": "Use composed prompts (e.g., p1 = a0*p0 + a_new*p_new) or not"
        }
    )
    detach_w_from_task_loss: bool = field(
        default=False,
        metadata={
            "help": "Optimize task-specific keys only based on query-key matching and detach the key optimization from the task loss"
        }
    )
    query_encoder_type: str = field(
        default="avg_all_embed",
        metadata={
            "help": "Embedding types, options: 'cls_token_embed', 'avg_all_embed' or 'avg_word_embed'"
        }
    ) 
    attn_init_func: str = field(
        default="uniform",
        metadata={
            "help": "In the compositional prompts setting, initialize attention filter vectors with 'orthogonal' / 'uniform' nn.init functions "
        }
    )
    key_init_func: str = field(
        default="uniform",
        metadata={
            "help": "In the compositional prompts setting, initialize key vectors with 'orthogonal' / 'uniform' nn.init functions "
        }
    )
    prompt_init_func: str = field(
        default="uniform",
        metadata={
            "help": "In the compositional prompts setting, initialize prompt vectors with 'orthogonal' / 'uniform' nn.init functions "
        }
    )
    general_prompt_init_func: str = field(
        default=None,
        metadata={
            "help": "In the compositional prompts setting, initialize general prompt vectors with 'orthogonal' / 'uniform' nn.init functions "
        }
    )
    ortho_loss_key: bool = field(
        default=False,
        metadata={
            "help": "In the compositional prompts setting, add orthogonal loss/penalty of keys or not"
        }
    ) 
    ortho_loss_prompt: bool = field(
        default=False,
        metadata={
            "help": "In the compositional prompts setting, add orthogonal loss/penalty of prompts or not"
        }
    ) 
    ortho_loss_coeff: float = field(
        default=0.1,
        metadata={
            "help": "In the compositional prompts setting, coefficient for the orthogonal loss/penalty term"
        }
    ) 
    direct_compose: bool = field(
        default=False,
        metadata={
            "help": "Directly use query-key cosine_similarity for prompt composition and matching loss calculation (if required). TODO: Make it a default setting"
        }
    ) 
    softmax_match_scale: int = field(
        default=8,
        metadata={
            "help": "Use scaled softmax given the cosine simialrity as input"
        }
    )
    matching_loss: bool = field(
        default=False,
        metadata={
            "help": "In the compositional prompts setting, add query <-> key matching loss or not"
        }
    ) 
    matching_loss_v2: bool = field(
        default=False,
        metadata={
            "help": "In the compositional prompts setting, add query <-> key matching loss or not, v2 uses the query-key distance while v1 uses matching probablity"
        }
    ) 
    matching_loss_cls: bool = field(
        default=False,
        metadata={
            "help": "In the compositional prompts setting, add query <-> key (classification) matching loss or not"
        }
    ) 
    matching_loss_cls_all: bool = field(
        default=False,
        metadata={
            "help": "In the compositional prompts setting, add query <-> key (classification) matching loss or not, consider all keys"
        }
    ) 
    matching_loss_coeff: float = field(
        default=1,
        metadata={
            "help": "In the compositional prompts setting, coefficient for the matching loss term"
        }
    )
    vanilla_continual_learning: bool = field(
        default=False,
        metadata={
            "help": "continuous produce new prompts based on previous prefix and past_key_value" # TODO: rewrite running code, make it real continuous
        }
    )
    cl_language_list: str = field(
        default='am_dz_ha_ig_kr_ma_pcm_pt_sw_ts_twi_yo',
        metadata={
            "help": "Language list in the continual learning mode, order matters"
        }
    )
    mtl_task_list: str = field(
        default='agnews_yelp_amazon_yahoo_dbpedia',
        metadata={
            "help": "Task list in the continual learning mode, order matters (BERT order-0 by default)"
        }
    )
    concat_prompts: bool = field(
        default=False,
        metadata={
            "help": "Progressively concatenate the prompt, 'False' means just using the current prompt"
        }
    )
    save_prompts: bool = field(
        default=False,
        metadata={
            "help": "Save prompts (task-specific and/or general) or not"
        }
    )
    save_initial_prompts: bool = field(
        default=False,
        metadata={
            "help": "Save initialized prompts before training starts (for comparison)"
        }
    )
    prompt_save_path: str = field(
        default=None,
        metadata={
            "help": "Path to the saved prompts"
        }
    )
    prompt_select_mode: str = field(
        default=None,
        metadata={
            "help": "Progressively concatenate the prompt, 'False' means just using the current prompt"
        }
    )
    task_specific_classifier: bool = field(
        default=False,
        metadata={
            "help": "Use task-specific classifier for each task (default case in ProgressivePrompt) or use a shared classifier"
        }
    )
    task_identify_epi: bool = field(
        default=False,
        metadata={
            "help": "Use task-specific classifier and select the most matching classifier based on Gaussian distribution prototypes during inference following EPI"
        }
    )
    epi_for_composition: bool = field(
        default=False,
        metadata={
            "help": "Use Gaussian distribution as task representations for module compositison"
        }
    )
    embed_prototypes_for_composition: bool = field(
        default=False,
        metadata={
            "help": "Use task embedding prototypes(mean) as task representations for module compositison"
        }
    )
    save_embed_statistics: bool = field(
        default=False,
        metadata={
            "help": "Save task-specific embedding statistics (mean and covariance of task embeddings) for further usage"
        }
    )
    save_embed_prototypes: bool = field(
        default=False,
        metadata={
            "help": "Save task-specific embedding prototypes (average embeddings) for further usage"
        }
    )
    classifier_match_embed: bool = field(
        default=False,
        metadata={
            "help": "Use task-specific classifier and select the most matching classifier based on embedding prototypes during inference"
        }
    )
    classifier_match_key: bool = field(
        default=False,
        metadata={
            "help": "Use task-specific classifier and select the most matching classifier based on task-specific keys during inference"
        }
    )
    compose_classifiers: bool = field(
        default=False,
        metadata={
            "help": "Compose (weighted sum) the prediction logits from different classifiers or not"
        }
    )
    complete_cil_eval: bool = field(
        default=False,
        metadata={
            "help": "Evaluate CIL results including CIL-direct & CIL-compose (besides CIL-top1) or not"
        }
    )
    prefix_hidden_size: int = field(
        default=512,
        metadata={
            "help": "The hidden size of the MLP projection head in Prefix Encoder if prefix projection is used"
        }
    )
    hidden_dropout_prob: float = field(
        default=0.1,
        metadata={
            "help": "The dropout probability used in the models"
        }
    )
    mpeft_enabled: bool = field(
        default=True,
        metadata={
            "help": "Enable 'mixture of peft modules' functions or not; If set to False, then just do the normal peft"
        }
    )

def get_args():
    """Parse all the args."""
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

    args = parser.parse_args_into_dataclasses()

    return args