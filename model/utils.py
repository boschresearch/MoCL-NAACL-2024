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

from enum import Enum
from model.sequence_classification import BertPrefixForSequenceClassification, RobertaPrefixForSequenceClassification
from model.sequence_classification_bert import BertPrefixContinualForSequenceClassification
from model.sequence_classification_roberta import RobertaPrefixContinualForSequenceClassification

from transformers import AutoConfig, RobertaForSequenceClassification


class TaskType(Enum):
    SEQUENCE_CLASSIFICATION = 2,

PREFIX_MODELS = {
    "bert": {
        TaskType.SEQUENCE_CLASSIFICATION: BertPrefixForSequenceClassification,
    },
    # "roberta": {
    #     TaskType.SEQUENCE_CLASSIFICATION: RobertaPrefixForSequenceClassification,
    # },
    "xlm-roberta": {
        TaskType.SEQUENCE_CLASSIFICATION: RobertaPrefixForSequenceClassification,
    },
}

MTL_MODELS = {
    "prefix":{
        "bert": {
            TaskType.SEQUENCE_CLASSIFICATION: BertPrefixContinualForSequenceClassification,
        },
        # "roberta": {
        #     TaskType.SEQUENCE_CLASSIFICATION: RobertaPrefixContinualForSequenceClassification
        # },
        "xlm-roberta": {
            TaskType.SEQUENCE_CLASSIFICATION: RobertaPrefixContinualForSequenceClassification
        },
    },
}

AUTO_MODELS = {
    # "roberta": {TaskType.SEQUENCE_CLASSIFICATION: RobertaForSequenceClassification,},
    "xlm-roberta": {TaskType.SEQUENCE_CLASSIFICATION: RobertaForSequenceClassification,},
}

def get_model(model_args, task_type: TaskType, config: AutoConfig, fix_bert: bool = False, language=None, seed=None, mtl=False, mtl5=False, max_seq_len=None, output_dir=None, label_list=None):
    for arg in dir(model_args):
        if not arg.startswith("__") and not callable(getattr(model_args, arg)):
            setattr(config, arg, getattr(model_args, arg))
    setattr(config, 'max_seq_length', max_seq_len)
    
    if model_args.prefix:
        config.language = language
        config.seed = seed
        config.output_dir = output_dir
        config.label_list = label_list
        config.mtl5 = mtl5
        
        if mtl:
            model_class = MTL_MODELS['prefix'][config.model_type][task_type]
            model = model_class.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                revision=model_args.model_revision,
            )
            if config.compose_prompts:
                try:
                    for param in model.query_encoder.parameters():
                        param.requires_grad = False
                except:
                    pass
        else:
            model_class = PREFIX_MODELS[config.model_type][task_type]
            model = model_class.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                revision=model_args.model_revision,
            )
    else:
        model_class = AUTO_MODELS[config.model_type][task_type]
        model = model_class.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            revision=model_args.model_revision,
        )

        bert_param = 0
        if fix_bert:
            if config.model_type == "bert":
                for param in model.bert.parameters():
                    param.requires_grad = False
                for _, param in model.bert.named_parameters():
                    bert_param += param.numel()
            elif config.model_type == "roberta":
                for param in model.roberta.parameters():
                    param.requires_grad = False
                for _, param in model.roberta.named_parameters():
                    bert_param += param.numel()
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
        total_param = all_param - bert_param
        print('***** total param is {} *****'.format(total_param))
    return model