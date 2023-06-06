# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Unimodal attention definition."""
import logging
from typing import Optional, Tuple
import torch
from typeguard import check_argument_types
from espnet.nets.pytorch_backend.transformer.attention import (
    LegacyRelPositionMultiHeadedAttention,
    MultiHeadedAttention,
    RelPositionMultiHeadedAttention,
)
from matplotlib import pyplot as plt
import librosa

class UMA(torch.nn.Module):
    """UMA module.

    Args:
    """

    def __init__(
        self,
        input_size: int = 256,
        output_size: int = 256,
    ):
        assert check_argument_types()
        super().__init__()
        self._output_size = output_size

        self.linear_sigmoid = torch.nn.Sequential(
            torch.nn.Linear(256, 1),
            torch.nn.Sigmoid(),
        )

    def output_size(self) -> int:
        return self._output_size

    def forward(
        self,
        xs_pad: torch.Tensor,
        olens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate forward propagation.

        Args:
            xs_pad (torch.Tensor): Input tensor (#batch, L, input_size).
            ilens (torch.Tensor): Input length (#batch).
            prev_states (torch.Tensor): Not to be used now.
        Returns:
            torch.Tensor: Output tensor (#batch, L, output_size).
            torch.Tensor: Output length (#batch).
            torch.Tensor: Not to be used now.
        """
        # Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]

        # masks = (~make_pad_mask(olens)[:, None, :]).to(xs_pad.device)
        # unimodal attention
        # plt.clf()
        # plt.plot(xs_pad.cpu().detach().numpy()[0,:,1], color='black')
        batch, length, _ = xs_pad.size()

        scalar_importance = self.linear_sigmoid(xs_pad)
        
        # plt.plot(scalar_importance.cpu().detach().numpy()[0,:,:], color='blue')
        # plt.savefig(f'./inference_images/hyp_{olens[0]}.png')
        # k = k+1
        
        # score = self.pooling_proj(xs_pad) / self._output_size**0.5 # (batch, T, 1)
        # scalar_importance = torch.softmax(score, dim=1)

        # print("alpha: ", scalar_importance)
        alpha_h = torch.mul(scalar_importance, xs_pad)

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

        utri_mat1 = torch.tril(torch.ones(max_counts+1,max_counts),-1).to(xs_pad.device)
        batch_index_mask = utri_mat1[counts]
        batch_index_mask = batch_index_mask.reshape(-1,1)
        batch_index_mask = batch_index_mask.nonzero()[:, 0]

        valleys = torch.zeros(batch * max_counts, 2).type_as(valley_index_start)
        valleys[batch_index_mask] = torch.cat((valley_index_start.unsqueeze(1), valley_index_end.unsqueeze(1)),1)
        # logging.info(str(valleys))
        
        # utri_mat = torch.tril(torch.cuda.FloatTensor(length+1,length).fill_(1),-1)
        utri_mat = torch.tril(torch.ones(length+1,length),-1).to(xs_pad.device)
        output_mask = (utri_mat[valleys[:,1]]-utri_mat[valleys[:,0]]).reshape(batch, max_counts, length)
        output_mask = output_mask.detach()

        xs_pad = torch.bmm(output_mask, alpha_h) / torch.bmm(output_mask, scalar_importance).clamp_(1e-6)
        # print(torch.isnan(output).any())
        
        # olens = (olens / olens[0] * xs_pad.shape[1]).type_as(olens)
        olens = counts
        
        # return xs_pad, olens, scalar_importance
        return xs_pad, olens
