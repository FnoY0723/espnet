#!/usr/bin/env python3
#  2021, University of Stuttgart;  Pavel Denisov
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Hugging Face Transformers PostEncoder."""

import copy
import logging
from typing import Tuple

import torch
from typeguard import check_argument_types

from espnet2.asr.postencoder.abs_postencoder import AbsPostEncoder
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.transformer.subsampling import TooShortUttError

try:
    from transformers import AutoModel

    is_transformers_available = True
except ImportError:
    is_transformers_available = False



class UnimodalAttentionPostEncoder(AbsPostEncoder):
    """Hugging Face Transformers PostEncoder."""

    def __init__(
        self,
        input_size: int,
        output_size: int = 256,
    ):
        """Initialize the module."""
        assert check_argument_types()
        super().__init__()

        self._output_size = output_size

        self.linear_sigmoid = torch.nn.Sequential(
            torch.nn.Linear(input_size, 1),
            torch.nn.Sigmoid(),
        )

    def forward(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]

        # masks = (~make_pad_mask(input_lengths)[:, None, :]).to(input.device)
        # unimodal attention
        # plt.clf()
        # plt.plot(input.cpu().detach().numpy()[0,:,1], color='black')
        batch, length, _ = input.size()

        scalar_importance = self.linear_sigmoid(input)
        
        # plt.plot(scalar_importance.cpu().detach().numpy()[0,:,:], color='blue')
        # plt.savefig(f'./inference_images/hyp_{input_lengths[0]}.png')
        # k = k+1
        
        # score = self.pooling_proj(input) / self._output_size**0.5 # (batch, T, 1)
        # scalar_importance = torch.softmax(score, dim=1)

        # print("alpha: ", scalar_importance)
        alpha_h = torch.mul(scalar_importance, input)

        # find valleys' index
        scalar_before = scalar_importance[:,:-1,:].detach()
        scalar_after = scalar_importance[:,1:,:].detach()
        scalar_before = torch.nn.functional.pad(scalar_before,(0,0,1,0))
        scalar_after = torch.nn.functional.pad(scalar_after,(0,0,0,1))

        mask = ((scalar_importance.lt(scalar_before)) & (scalar_importance.lt(scalar_after)))
        mask = mask.reshape(scalar_importance.shape[0], -1)
        # print(mask.shape)
        mask[:,0] = True
        batch_index = mask.nonzero()[:,0]
        valley_index_start = mask.nonzero()[:,1]
        mask[:,0] = False
        mask[:,-1] = True
        valley_index_end = valley_index_start + 2
        valley_index_end = torch.where(valley_index_end > (length) * torch.ones_like(valley_index_end), 
                                       (length) * torch.ones_like(valley_index_end), valley_index_end)
        # print(valley_index_start.shape)
        # print(valley_index_end.shape)

        _,counts = torch.unique(batch_index, return_counts = True)
        # logging.info(str(counts))
        max_counts = (torch.max(counts)).item()

        utri_mat1 = torch.tril(torch.ones(max_counts+1,max_counts),-1).to(input.device)
        batch_index_mask = utri_mat1[counts]
        batch_index_mask = batch_index_mask.reshape(-1,1)
        batch_index_mask = batch_index_mask.nonzero()[:, 0]

        valleys = torch.zeros(batch * max_counts, 2).type_as(valley_index_start)
        valleys[batch_index_mask] = torch.cat((valley_index_start.unsqueeze(1), valley_index_end.unsqueeze(1)),1)
        # logging.info(str(valleys))
        
        # utri_mat = torch.tril(torch.cuda.FloatTensor(length+1,length).fill_(1),-1)
        utri_mat = torch.tril(torch.ones(length+1,length),-1).to(input.device)
        output_mask = (utri_mat[valleys[:,1]]-utri_mat[valleys[:,0]]).reshape(batch, max_counts, length)
        output_mask = output_mask.detach()

        xs_pad = torch.bmm(output_mask, alpha_h) / torch.bmm(output_mask, scalar_importance).clamp_(1e-6)
        # print(torch.isnan(output).any())
        
        # olens = (input_lengths / input_lengths[0] * xs_pad.shape[1]).type_as(input_lengths)
        olens = counts
        
        # return xs_pad, olens, scalar_importance
        return xs_pad, olens
        

    def output_size(self) -> int:
        """Get the output size."""
        return self._output_size


