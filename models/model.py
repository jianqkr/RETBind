# Copyright (c) EPFL VILAB.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# Based on timm, DeiT, DINO, MoCo-v3, BEiT, MAE-priv and MAE code bases
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# https://github.com/facebookresearch/moco-v3
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/BUPT-PRIV/MAE-priv
# https://github.com/facebookresearch/mae
# --------------------------------------------------------
# Modified from EPFL-VILAB/MultiMAE for RETBind training.

import itertools
import math
from collections import OrderedDict
from functools import partial
from typing import Dict, List, Optional, Union

import torch
from einops import rearrange, repeat
from torch import nn
import torch.nn.functional as F
from torch.distributions.dirichlet import Dirichlet

from utils.registry import register_model

from .model_utils import Block, trunc_normal_, MaskedAttention, MaskedBlock

__all__ = [
    'pretrain_multimae_base',
    'pretrain_masked_multimae_base',
]


class MultiMAE(nn.Module):
    """MultiMAE: Multi-task Multi-modal Masked Autoencoder
    This module performs masking in its forward pass.
    The MultiViT module defined below inherits from this module and performs a regular forward pass,
    and should be used instead for downstream tasks


    :param input_adapters: Dictionary of task -> input adapters
    :param output_adapters: Optional dictionary of task -> output adapters

    :param num_global_tokens: Number of additional global tokens to add (like cls tokens), default is 1
    :param dim_tokens: Dimension of encoder tokens
    :param depth: Depth of encoder
    :param num_heads: Number of attention heads
    :param mlp_ratio: MLP hidden dim ratio
    :param qkv_bias: Set to False to disable bias
    :param drop_rate: Dropout after MLPs and Attention
    :param attn_drop_rate: Attention matrix drop rate
    :param drop_path_rate: DropPath drop rate
    :param norm_layer: Type of normalization layer
    """
    def __init__(self,
                 input_adapters: Dict[str, nn.Module],
                 output_adapters: Optional[Dict[str, nn.Module]],
                 num_global_tokens: int = 1,
                 use_modality_tokens: bool = True,
                 use_masked_attention: bool = True,
                 dim_tokens: int = 768,
                 depth: int = 12,
                 num_heads: int = 12,
                 mlp_ratio: float = 4.0,
                 qkv_bias: bool = True,
                 drop_rate: float = 0.0,
                 attn_drop_rate: float = 0.0,
                 drop_path_rate: float = 0.0,
                 norm_layer: nn.Module = partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()

        # Initialize input and output adapters
        for adapter in input_adapters.values():
            adapter.init(dim_tokens=dim_tokens)
        self.input_adapters = nn.ModuleDict(input_adapters)
        if output_adapters is not None:
            for adapter in output_adapters.values():
                adapter.init(dim_tokens_enc=dim_tokens)
            self.output_adapters = nn.ModuleDict(output_adapters)
        else:
            self.output_adapters = None

        # Additional learnable tokens that can be used by encoder to process/store global information
        self.num_global_tokens = num_global_tokens
        self.global_tokens = nn.Parameter(torch.zeros(1, num_global_tokens, dim_tokens))
        trunc_normal_(self.global_tokens, std=0.02)

        # Add modality-specific tokens
        self.use_modality_tokens = use_modality_tokens
        if use_modality_tokens:
            self.modality_tokens = nn.ParameterDict({
                modality: nn.Parameter(torch.zeros(1, 1, dim_tokens))
                for modality in input_adapters.keys()
            })
            # Initialize with trunc_normal
            for token in self.modality_tokens.values():
                trunc_normal_(token, std=0.02)

        # Transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        if use_masked_attention:
            self.encoder = nn.ModuleList([
                MaskedBlock(dim=dim_tokens, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                      drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
                for i in range(depth)
            ])
        else:
            self.encoder = nn.Sequential(*[
                Block(dim=dim_tokens, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                      drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
                for i in range(depth)
            ])

        self.apply(self._init_weights)
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                if 'qkv' in name:
                    val = math.sqrt(6. / float(m.weight.shape[0] // 3 + m.weight.shape[1]))
                    nn.init.uniform_(m.weight, -val, val)
                elif 'kv' in name:
                    val = math.sqrt(6. / float(m.weight.shape[0] // 2 + m.weight.shape[1]))
                    nn.init.uniform_(m.weight, -val, val)

            if isinstance(m, nn.Conv2d):
                if '.proj' in name:
                    w = m.weight.data
                    nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        self.use_masked_attention = use_masked_attention

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.encoder)

    @torch.jit.ignore
    def no_weight_decay(self):
        no_wd_set = {'global_tokens'}

        if self.use_modality_tokens:
            for modality in self.modality_tokens:
                no_wd_set.add(f'modality_tokens.{modality}')

        for task, adapter in self.input_adapters.items():
            if hasattr(adapter, 'no_weight_decay'):
                to_skip = adapter.no_weight_decay()
                to_skip = set([f'input_adapters.{task}.{name}' for name in to_skip])
                no_wd_set = no_wd_set | to_skip

        if self.output_adapters is not None:
            for task, adapter in self.output_adapters.items():
                if hasattr(adapter, 'no_weight_decay'):
                    to_skip = adapter.no_weight_decay()
                    to_skip = set([f'output_adapters.{task}.{name}' for name in to_skip])
                    no_wd_set = no_wd_set | to_skip

        return no_wd_set

    def sample_alphas(self, B: int, n_tasks: int, alphas: float = 1.0, eps: float = 1e-5):
        """
        Sample alphas for Dirichlet sampling such that tasks are first uniformly chosen and then Dirichlet sampling
        is performed over the chosen ones.

        :param B: Batch size
        :param n_tasks: Number of input tasks
        :param alphas: Float or list to multiply task choices {0,1} by
        :param eps: Small constant since Dirichlet alphas need to be positive
        """
        valid_task_choices = torch.Tensor([list(i) for i in itertools.product([0, 1], repeat=n_tasks)][1:])
        rand_per_sample_choice = torch.randint(0, len(valid_task_choices), (B,))
        alphas_tensor = torch.index_select(valid_task_choices, 0, rand_per_sample_choice)
        alphas_tensor = alphas_tensor * torch.tensor(alphas) + eps
        return alphas_tensor

    def generate_random_masks(self,
                            input_tokens: Dict[str, torch.Tensor],
                            num_encoded_tokens: int,
                            alphas: Union[float, List[float]] = 1.0,
                            sample_tasks_uniformly: bool = False) :
        """
        Sample a total of num_encoded_tokens from different tasks using Dirichlet sampling.

        :param input_tokens: Dictionary of tensors to sample num_encoded_tokens from
        :param num_encoded_tokens: Number of tokens to select
        :param alphas: Dirichlet distribution parameter alpha. Lower alpha = harder,
            less uniform sampling. Can be float or list of floats.
        :param sample_tasks_uniformly: Set to True to first sample 1-n_tasks uniformly at random
            for each sample in the batch. Dirichlet sampling is then done over selected subsets.
        """
        B = list(input_tokens.values())[0].shape[0]
        device = list(input_tokens.values())[0].device

        alphas = [alphas] * len(input_tokens) if isinstance(alphas, float) else alphas
        if sample_tasks_uniformly:
            alphas = self.sample_alphas(B, len(input_tokens), alphas=alphas)
            task_sampling_dist = Dirichlet(alphas).sample().to(device)
        else:
            task_sampling_dist = Dirichlet(torch.Tensor(alphas)).sample((B,)).to(device)

        num_domains = len(input_tokens)
        remaining   = num_encoded_tokens - num_domains
        samples_extra = (task_sampling_dist * remaining).round().long().clamp(min=0)

        total = samples_extra.sum(dim=1)        # [B]
        diff  = remaining - total               # [B]
        if (diff != 0).any():
            adjust_idx = samples_extra.argmax(dim=1)
            samples_extra.scatter_add_(1, adjust_idx.unsqueeze(1), diff.unsqueeze(1))
            samples_extra = samples_extra.clamp(min=0)

        samples_per_task = samples_extra + 1  

        task_masks = []
        num_tokens_per_task = [task_tokens.shape[1] for task_tokens in input_tokens.values()]

        for i, num_tokens in enumerate(num_tokens_per_task):
            noise = torch.rand(B, num_tokens, device=device)
            ids_arange_shuffle = torch.argsort(noise, dim=1)
            mask = torch.arange(num_tokens, device=device).unsqueeze(0).expand(B, -1)
            mask = torch.gather(mask, dim=1, index=ids_arange_shuffle)
            mask = torch.where(mask < samples_per_task[:, i].unsqueeze(1), 0, 1)
            task_masks.append(mask)

        mask_all = torch.cat(task_masks, dim=1)
        ids_shuffle = torch.argsort(mask_all + torch.rand_like(mask_all.float()), dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :num_encoded_tokens]

        mask_all = torch.ones_like(mask_all)
        mask_all[:, :num_encoded_tokens] = 0
        mask_all = torch.gather(mask_all, dim=1, index=ids_restore)
        task_masks = torch.split(mask_all, num_tokens_per_task, dim=1)
        task_masks = {domain: mask for domain, mask in zip(input_tokens.keys(), task_masks)}

        return task_masks, ids_keep, ids_restore

    def create_attention_mask(self,
                              domain_label: torch.Tensor,
                              num_patch_tokens: int,
                              num_modality_tokens: int,
                              B: int,
                              N: int) -> torch.Tensor:
        
        device = self.global_tokens.device
        mask = torch.zeros(B, N, N, device=device)

        same_domain = (domain_label.unsqueeze(2) == domain_label.unsqueeze(1)).float()
        mask[:, :num_patch_tokens, :num_patch_tokens] = same_domain

        for d_idx in range(num_modality_tokens):
            mod_pos = num_patch_tokens + d_idx
            has_domain = (domain_label == d_idx).any(dim=1).float()  
            in_domain  = (domain_label == d_idx).float()        
            weighted   = in_domain * has_domain.unsqueeze(1)      
            mask[:, :num_patch_tokens, mod_pos] = weighted
            mask[:, mod_pos, :num_patch_tokens] = weighted

        mod_start = num_patch_tokens
        mod_end   = num_patch_tokens + num_modality_tokens
        mask[:, mod_start:mod_end, mod_start:mod_end] = 1.0

        mask[:, mod_end:, :] = 1.0 
        mask[:, :, mod_end:] = 1.0 

        return mask

    @staticmethod
    def make_mask(N_H, N_W, xy_idxs, full_tasks=[], indicate_visible=True, flatten=True, device='cuda'):
        """
        Creates masks for each task, given lists of un-masked x,y coordinates.
        """
        xy_idxs = {
            k: torch.LongTensor(v)
            for k, v in xy_idxs.items()
        }

        task_masks = {
            k: torch.ones(N_H, N_W).to(device)
            for k in xy_idxs.keys()
        }

        for k in xy_idxs.keys():
            if len(xy_idxs[k]) > 0:
                task_masks[k][xy_idxs[k][:, 1], xy_idxs[k][:, 0]] = 0

        for task in full_tasks:
            task_masks[task][:] = 0

        if not indicate_visible:
            task_masks = {k: 1 - v for k, v in task_masks.items()}

        if flatten:
            task_masks = {k: v.flatten().unsqueeze(0) for k, v in task_masks.items()}

        return task_masks

    def generate_input_info(self, input_task_tokens, image_size):
        input_info = OrderedDict()
        i = 0
        input_info['tasks'] = {}
        for domain, tensor in input_task_tokens.items():
            num_tokens = tensor.shape[1]
            d = {
                'num_tokens': num_tokens,
                'has_2d_posemb': True,
                'start_idx': i,
                'end_idx': i + num_tokens,
            }
            i += num_tokens
            input_info['tasks'][domain] = d

        input_info['image_size'] = image_size
        input_info['num_task_tokens'] = i
        input_info['num_global_tokens'] = self.num_global_tokens
        input_info['num_modality_tokens'] = len(input_task_tokens)

        return input_info

    def forward(self,
                x: Union[Dict[str, torch.Tensor], torch.Tensor],
                mask_inputs: bool = True,
                task_masks: Dict[str, torch.Tensor] = None,
                num_encoded_tokens: int = 128,
                alphas: Union[float, List[float]] = 1.0,
                sample_tasks_uniformly: bool = False,
                fp32_output_adapters: List[str] = []):
        """
        Forward pass through input adapters, transformer encoder and output adapters.
        If specified, will randomly drop input tokens.
        """
        ## Processing input modalities
        B = next(iter(x.values())).shape[0]

        assert all(k in x for k in ['oct', 'uwf']), 'Expected keys: oct, uwf'

        _, _, H, W = x['uwf'].shape

        # Encode selected inputs to tokens
        input_task_tokens = {}
        for domain in ['oct','uwf']:
            patch_tokens = self.input_adapters[domain](x[domain])  # [B, N, D]
            input_task_tokens[domain] = patch_tokens

        input_info = self.generate_input_info(input_task_tokens=input_task_tokens, image_size=(H, W))

        # Select random subset of tokens from the chosen input tasks and concatenate them
        if mask_inputs:
            num_encoded_tokens = num_encoded_tokens if num_encoded_tokens is not None else self.num_encoded_tokens
        else:
            num_encoded_tokens = sum([tensor.shape[1] for tensor in input_task_tokens.values()])

        if task_masks is None:
            task_masks, ids_keep, ids_restore = self.generate_random_masks(
                input_task_tokens, num_encoded_tokens, alphas=alphas, sample_tasks_uniformly=sample_tasks_uniformly
            )

        else:
            mask_all = torch.cat([task_masks[task] for task in input_task_tokens.keys()], dim=1)
            ids_shuffle = torch.argsort(mask_all, dim=1)
            ids_restore = torch.argsort(ids_shuffle, dim=1)
            ids_keep = ids_shuffle[:, :num_encoded_tokens]

        input_tokens = torch.cat([task_tokens for task_tokens in input_task_tokens.values()], dim=1)
        input_tokens = torch.gather(input_tokens, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, input_tokens.shape[2]))

        if self.use_masked_attention:
            domain_names = list(input_task_tokens.keys())
            boundaries = [0]
            for tokens in input_task_tokens.values():
                boundaries.append(boundaries[-1] + tokens.shape[1])
            np_tokens = input_tokens.shape[1]  
            domain_label = torch.zeros(B, np_tokens, dtype=torch.long, device=input_tokens.device)
            for d_idx in range(len(domain_names)):
                in_domain = (ids_keep >= boundaries[d_idx]) & (ids_keep < boundaries[d_idx + 1])
                domain_label[in_domain] = d_idx

        if self.use_modality_tokens:
            modality_tokens_list = []
            for domain in input_task_tokens.keys():
                visible_count = (task_masks[domain] == 0).sum(dim=1)  # [B]
                modality_mask = (visible_count > 0).unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1]
                mod_token = self.modality_tokens[domain].expand(B, 1, -1)  # [B, 1, D]
                mod_token = mod_token * modality_mask
                modality_tokens_list.append(mod_token)

            modality_tokens = torch.cat(modality_tokens_list, dim=1)  # [B, num_modalities, D]
            input_tokens = torch.cat([input_tokens, modality_tokens], dim=1)

        global_tokens = repeat(self.global_tokens, '() n d -> b n d', b=B)
        input_tokens = torch.cat([input_tokens, global_tokens], dim=1)

        if self.use_masked_attention:
            N = input_tokens.shape[1]
            num_mod = len(input_task_tokens) if self.use_modality_tokens else 0
            attention_mask = self.create_attention_mask(
                domain_label, np_tokens, num_mod, B, N
            )
            x = input_tokens
            for block in self.encoder:
                x = block(x, attention_mask)
            encoder_tokens = x
        else:
            encoder_tokens = self.encoder(input_tokens)

        if self.output_adapters is None:
            return encoder_tokens, task_masks

        if self.use_modality_tokens:
            input_info['num_modality_tokens'] = len(input_task_tokens)

        preds = {
            domain: self.output_adapters[domain](
                encoder_tokens=encoder_tokens,
                input_info=input_info,
                ids_keep=ids_keep,
                ids_restore=ids_restore,
            )
            for domain in self.output_adapters
            if domain not in fp32_output_adapters
        }

        with torch.amp.autocast('cuda', enabled=False):
            for domain in fp32_output_adapters:
                if domain not in self.output_adapters:
                    continue
                preds[domain] = self.output_adapters[domain](
                    encoder_tokens=encoder_tokens.float(),
                    input_info=input_info,
                    ids_keep=ids_keep,
                    ids_restore=ids_restore,
                )

        return preds, task_masks

@register_model
def pretrain_multimae_base(
        input_adapters: Dict[str, nn.Module],
        output_adapters: Optional[Dict[str, nn.Module]],
        **kwargs):
    model = MultiMAE(
        input_adapters=input_adapters,
        output_adapters=output_adapters,
        use_modality_tokens=kwargs.pop('use_modality_tokens', True),
        use_masked_attention=kwargs.pop('use_masked_attention', False),
        dim_tokens=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model

@register_model
def pretrain_masked_multimae_base(
        input_adapters: Dict[str, nn.Module],
        output_adapters: Optional[Dict[str, nn.Module]],
        **kwargs):
    model = MultiMAE(
        input_adapters=input_adapters,
        output_adapters=output_adapters,
        use_modality_tokens=kwargs.pop('use_modality_tokens', True),
        dim_tokens=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model
