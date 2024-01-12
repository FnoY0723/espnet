
'''
Author: FnoY fangying@westlake.edu.cn
LastEditTime: 2024-01-11 21:34:41
FilePath: /espnet/espnet2/asr/uma_att.py
Notes: If the feature dimension changes from 256 to 512, just modify 'output_size: int = 256' to 'output_size: int = 512'.
'''
# """Unimodal aggregation definition."""
import logging
from typing import Optional, Tuple
import torch
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from typeguard import check_argument_types
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
import math

class UMA(torch.nn.Module):
    """UMA module.

    """

    def __init__(
        self,
        input_size: int = 256,
        output_size: int = 256,
        dropout_rate: float = 0.0,
    ):
        assert check_argument_types()
        super().__init__()
        self._output_size = output_size
        input_size = output_size

        # self.linear_sigmoid = torch.nn.Sequential(
        #     torch.nn.Linear(input_size, 1),
        #     torch.nn.Sigmoid(),
        # )
        
        self.gaussian_w = torch.nn.Parameter(torch.tensor(1e-10),requires_grad=True)
        # self.gaussian_b = None
        self.gaussian_bias = None
        # self.gaussian_w = 0.005
        # self.gaussian_b = -0.05
        # self.gaussian_w = 0.005
        # self.gaussian_b = -0.0

        self.h = 1
        self.d_k = input_size
        self.linear_q = torch.nn.Linear(input_size, input_size)
        self.linear_k = torch.nn.Linear(input_size, input_size)
        self.linear_v = torch.nn.Linear(input_size, input_size)
        # self.linear_out = torch.nn.Linear(input_size, input_size)
        self.attn = None
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        # self.avepool = torch.nn.AvgPool1d(3, stride=1, padding=1)

    def output_size(self) -> int:
        return self._output_size

    def forward_qkv(self, query, key, value):
        """Transform query, key and value.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).

        Returns:
            torch.Tensor: Transformed query tensor (#batch, n_head, time1, d_k).
            torch.Tensor: Transformed key tensor (#batch, n_head, time2, d_k).
            torch.Tensor: Transformed value tensor (#batch, n_head, time2, d_k).

        """
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
        # q = k
        q = q.transpose(1, 2)  # (batch, head, time1, d_k)
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)

        return q, k, v
    

    def forward(
        self,
        xs_pad: torch.Tensor,
        olens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Calculate forward propagation.

        Args:
            xs_pad (torch.Tensor): Input tensor (#batch, L, input_size).
            olens (torch.Tensor): Input length (#batch).
            prev_states (torch.Tensor): Not to be used now.
        Returns:
            torch.Tensor: Output tensor (#batch, I, output_size).
            torch.Tensor: Output length (#batch).
            torch.Tensor: Not to be used now.
        """
        masks = (~make_pad_mask(olens)[:, None, :]).to(xs_pad.device)
        # logging.info(olens) 
        
        batch, length, _ = xs_pad.size()

        q, k, v = self.forward_qkv(xs_pad, xs_pad, xs_pad) # (batch, time2, size)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        # scores = torch.matmul(q, k.transpose(-2, -1))
        
        # qk_mask = (torch.tril(torch.ones(length,length), diagonal=5) - 
        #             torch.tril(torch.ones(length,length),diagonal=-5)).to(xs_pad.device).reshape(1,length,length)
        gaussian_map = (torch.arange(0, length)).repeat(length, 1).to(xs_pad.device)
        distance_squre = torch.square(gaussian_map - gaussian_map.T)

        # logging.info(f'gaussian_w: {self.gaussian_w}, gaussian_b: {self.gaussian_b}')
        self.gaussian_bias = torch.abs(self.gaussian_w * distance_squre)
        # self.gaussian_bias = torch.abs(self.gaussian_w * distance_squre + self.gaussian_b)
        scores = scores - self.gaussian_bias
        # logging.info(qk_mask.shape, scores.shape)
        # scores = torch.bmm(qk_mask, scores.squeeze(1))

        if masks is not None:
            masks = masks.unsqueeze(1).eq(0)  # (batch, 1, *, time2)
            min_value = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(masks, min_value)
            self.attn = torch.softmax(scores, dim=-1).masked_fill(
                masks, 0.0
            )  # (batch, head, time1, time2)
        else:
            self.attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)
        
        p_attn = self.dropout(self.attn)

        uma_weights = torch.sum(p_attn, dim=-2).view(batch, length, 1)

        # median filter for uma_weights
        # uma_weights_all = torch.cat([torch.nn.functional.pad(uma_weights_origin[:,:-1,:],(0,0,1,0)), 
        #                          uma_weights_origin, 
        #                          torch.nn.functional.pad(uma_weights_origin[:,1:,:],(0,0,0,1))], -1)
        # uma_weights,_ = torch.median(uma_weights_all,-1)
        # uma_weights = uma_weights.view(batch, length, 1)

        # average filter for uma_weights
        # uma_weights = (self.avepool(uma_weights_origin.permute(0,2,1))).permute(0,2,1)

        # Use Linear-Sigmoid to generate unimodal aggregation weights
        # uma_weights: (#batch, L, 1)
        # uma_weights = self.linear_sigmoid(xs_pad)

        # Unimodal Detection
        scalar_before = uma_weights[:,:-1,:].detach() # (#batch, L-1, 1)
        scalar_after = uma_weights[:,1:,:].detach() # (#batch, L-1, 1)
        scalar_before = torch.nn.functional.pad(scalar_before,(0,0,1,0))    # (#batch, L, 1)
        scalar_after = torch.nn.functional.pad(scalar_after,(0,0,0,1))  # (#batch, L, 1)

        mask = (uma_weights.lt(scalar_before)) & (uma_weights.lt(scalar_after)) # bool tensor (#batch, L, 1)
        mask = mask.reshape(uma_weights.shape[0], -1) # bool tensor (#batch, L)
        mask[:,0] = True 
        # mask.nonzero() is [[0,0],[0,3],[0,7],...,[1,0],[1,2],...,[2,0],[2,4],...,[#batch-1,0],...]
        # mask.nonzero() : (K,2); K is the total number of valleys in this batch
        batch_index = mask.nonzero()[:,0] # (k,1); [0,0,0,...,1,1,...,2,2,...,#batch-1,...]
        valley_index_start = mask.nonzero()[:,1] # (k,1); [0,3,7,...,0,2,...,0,4,...,0,...]
        mask[:,0] = False
        mask[:,-1] = True
        valley_index_end = mask.nonzero()[:,1] + 2 
        # (k,1); [5,9,...,4,...,6,...]
        valley_index_end = torch.where(valley_index_end > (length) * torch.ones_like(valley_index_end), 
                                       (length) * torch.ones_like(valley_index_end), valley_index_end)

        _,counts = torch.unique(batch_index, return_counts = True) # (#batch, 1); the number of valleys in each sample
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

        # Aggregation
        # alpha_h = torch.mul(uma_weights_origin, xs_pad)
        # xs_pad = torch.bmm(output_mask, alpha_h) / torch.bmm(output_mask, uma_weights_origin).clamp_(1e-6)
        alpha_h = torch.mul(uma_weights, xs_pad)
        xs_pad = torch.bmm(output_mask, alpha_h) / torch.bmm(output_mask, uma_weights).clamp_(1e-6)
 
        # olens = (olens / olens[0] * xs_pad.shape[1]).type_as(olens)
        olens = counts

        # return xs_pad, olens, (uma_weights, p_attn)
        return xs_pad, olens, (self.gaussian_w, None)
        # return xs_pad, olens, uma_weights
        # return xs_pad, olens, None
