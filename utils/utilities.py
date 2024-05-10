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

import torch

def get_valid_word_embed(embeddings, attention_masks):
    embeddings_valid = torch.zeros_like(embeddings[:, 0, :])
    batch_size = embeddings.shape[0]
    for i in range(batch_size):
        n_valid_tokens = sum(attention_masks[i])
        embedding = embeddings[i][1:n_valid_tokens-1] # remove [cls], [end] and all padding tokens
        embeddings_valid[i] = torch.mean(embedding, dim=0)
    return embeddings_valid


def mahalanobis(querys, mean, cov_inv, norm=2):
    """
    args:
        querys: [n, dim]
        mean: [dim]
        cov_inv: [dim, dim]
    return：
        [n]
    """
    diff = querys - mean
    # [n, dim] = ([n, dim] @ [dim, dim]) * [n, dim] = [n, dim] * [n, dim]
    maha_dis = torch.matmul(diff, cov_inv.cuda()) * diff

    if norm == 2:
        return maha_dis.sum(dim=1)
    if norm == 1:
        return maha_dis.abs().sqrt().sum(dim=1)
    if norm == 'inf':
        return maha_dis.max(dim=1)


def print_and_log(new_text, log_text):
    print(new_text)
    log_text += new_text + "\n"
    return log_text


def mask_labels(inputs, label_token_id, pad_token_id, split=False):
    seq_len = inputs.shape[-1]
    inputs = inputs.tolist()
    masked_inputs = []
    input_len = []
    for idx, ids in enumerate(inputs):
        # if label_token_id in ids:
        label_token_index = ids.index(label_token_id)
        input_len.append(label_token_index)
        
        text_ids = ids[:label_token_index+1]
        if not split:
            text_ids = pad_to_max_len(text_ids, seq_len-len(text_ids), pad_token_id).tolist()
        # else:
        #     text_ids = ids
        masked_inputs.append(text_ids)
    
    return torch.tensor(masked_inputs), torch.tensor(input_len)

    
def pad_to_max_len(l, pad_len, val, left=False):
    if type(l) != list:
        l = l.tolist()
    padded = [val] * pad_len + l if left else l + [val] * pad_len
    
    return torch.tensor(padded)
    
    