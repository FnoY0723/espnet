# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Unimodal attention definition."""
import logging
from typing import Optional, List, Union, Tuple
import torch
from typeguard import check_argument_types

from espnet2.asr.ctc import CTC
from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.layers.cgmlp import ConvolutionalGatingMLP
from espnet.nets.pytorch_backend.conformer.convolution import ConvolutionModule
from espnet.nets.pytorch_backend.conformer.encoder_layer import EncoderLayer
from espnet.nets.pytorch_backend.nets_utils import get_activation, make_pad_mask
from espnet.nets.pytorch_backend.transformer.attention import (
    LegacyRelPositionMultiHeadedAttention,
    MultiHeadedAttention,
    RelPositionMultiHeadedAttention,
)
from espnet.nets.pytorch_backend.transformer.embedding import (
    LegacyRelPositionalEncoding,
    PositionalEncoding,
    RelPositionalEncoding,
    ScaledPositionalEncoding,
)
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet.nets.pytorch_backend.transformer.multi_layer_conv import (
    Conv1dLinear,
    MultiLayeredConv1d,
)
from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import (
    PositionwiseFeedForward,
)
from espnet.nets.pytorch_backend.transformer.repeat import repeat
from espnet.nets.pytorch_backend.transformer.subsampling import (
    Conv2dSubsampling,
    Conv2dSubsampling1,
    Conv2dSubsampling2,
    Conv2dSubsampling6,
    Conv2dSubsampling8,
    TooShortUttError,
    check_short_utt,
)


class UMA(torch.nn.Module):
    """UMA module.

    Args:
    """

    def __init__(
        self,
        input_size: int = 256,
        output_size: int = 256,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.0,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        input_layer: str = "conv2d",
        normalize_before: bool = True,
        concat_after: bool = False,
        positionwise_layer_type: str = "linear",
        positionwise_conv_kernel_size: int = 3,
        macaron_style: bool = False,
        rel_pos_type: str = "legacy",
        pos_enc_layer_type: str = "rel_pos",
        selfattention_layer_type: str = "rel_selfattn",
        activation_type: str = "swish",
        use_cnn_module: bool = True,
        zero_triu: bool = False,
        cnn_module_kernel: int = 31,
        padding_idx: int = -1,
        interctc_layer_idx: List[int] = [],
        interctc_use_conditioning: bool = False,
        stochastic_depth_rate: Union[float, List[float]] = 0.0,
        layer_drop_rate: float = 0.0,
        max_pos_emb_len: int = 5000,
        uma_method: str = "linear_sigmoid", #linear_sigmoid; att_pooling; att1; att2; cg_mlp
        cgmlp_linear_units: int = 2048,
        cgmlp_conv_kernel: int = 31,
        use_linear_after_conv: bool = False,
        gate_activation: str = "identity",
    ):
        assert check_argument_types()
        super().__init__()
        self._output_size = output_size

        self.uma_method = uma_method
        activation = get_activation(activation_type)
        pos_enc_class = RelPositionalEncoding

        # self.embed = torch.nn.Sequential(
        #     torch.nn.Linear(input_size, output_size),
        #     torch.nn.LayerNorm(output_size),
        #     torch.nn.Dropout(dropout_rate),
        #     pos_enc_class(output_size, positional_dropout_rate, max_pos_emb_len),
        # )

        # cgmlp_layer = ConvolutionalGatingMLP
        # cgmlp_layer_args = (
        #     output_size,
        #     cgmlp_linear_units,
        #     cgmlp_conv_kernel,
        #     dropout_rate,
        #     use_linear_after_conv,
        #     gate_activation,
        # )
        
        # self.cgmlp = cgmlp_layer(*cgmlp_layer_args)
        # #cg_mlp
        # self.norm_mlp = LayerNorm(input_size)
        # # encoder_selfattn_layer = RelPositionMultiHeadedAttention
        # # encoder_selfattn_layer_args = (
        # #     attention_heads,
        # #     output_size,
        # #     attention_dropout_rate,
        # #     zero_triu,
        # # )

        # # # Attention pooling
        # # self.pooling_proj = torch.nn.Linear(256, 1)

        # # self.norm_mha = LayerNorm(input_size)
        # # self.attn = encoder_selfattn_layer(*encoder_selfattn_layer_args)
        # # # positionwise_layer(*positionwise_layer_args)
            
        # # self.after_norm = LayerNorm(output_size)
        # self.dropout = torch.nn.Dropout(dropout_rate)
        
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

        if self.uma_method == "att1":
            masks = (~make_pad_mask(olens)[:, None, :]).to(xs_pad.device)
            # unimodal attention
            xs_pad = self.embed(xs_pad)

            xs_pad, pos_emb = xs_pad[0], xs_pad[1]
            xs_pad = self.norm_mha(xs_pad)
            x_att = self.attn(xs_pad, xs_pad, xs_pad, pos_emb, masks)
            xs_pad = x_att
            # xs_pad = self.dropout(x_att)

            if isinstance(xs_pad, tuple):
                xs_pad = xs_pad[0]

            xs_pad = self.after_norm(xs_pad)
            olens = masks.squeeze(1).sum(1)
            batch, length, _ = xs_pad.size()
            scalar_importance = self.linear_sigmoid(xs_pad) 
        
        elif self.uma_method == "att2":
            masks = (~make_pad_mask(olens)[:, None, :]).to(xs_pad.device)
            xs_pad1 = self.embed(xs_pad)

            xs_pad1, pos_emb = xs_pad1[0], xs_pad1[1]
            xs_pad1 = self.norm_mha(xs_pad1)
            x_att = self.attn(xs_pad1, xs_pad1, xs_pad1, pos_emb, masks)
            xs_pad1 = x_att
            # xs_pad = self.dropout(x_att)

            if isinstance(xs_pad1, tuple):
                xs_pad1 = xs_pad1[0]

            xs_pad1 = self.after_norm(xs_pad1)
            olens = masks.squeeze(1).sum(1)
            batch, length, _ = xs_pad.size()
            scalar_importance = self.linear_sigmoid(xs_pad)
        #############################################################################################
        elif self.uma_method == "linear_sigmoid":
            batch, length, _ = xs_pad.size()
            scalar_importance = self.linear_sigmoid(xs_pad)
        
        elif self.uma_method == "att_pooling":
            # att_pooling
            score = self.pooling_proj(xs_pad) / self._output_size**0.5 # (batch, T, 1)
            scalar_importance = torch.softmax(score, dim=1)
        
        elif self.uma_method == "cg_mlp":
            masks = (~make_pad_mask(olens)[:, None, :]).to(xs_pad.device)

            xs_pad = self.norm_mlp(xs_pad)
            xs_pad = self.cgmlp(xs_pad, masks)
            if isinstance(xs_pad, tuple):
                xs_pad = xs_pad[0]

            xs_pad = self.dropout(xs_pad)
            batch, length, _ = xs_pad.size()
            scalar_importance = self.linear_sigmoid(xs_pad)
        
        # print("alpha: ", scalar_importance)
        alpha_h = torch.mul(scalar_importance, xs_pad)

        # find valleys' index
        scalar_before = scalar_importance[:,:-1,:].detach()
        scalar_after = scalar_importance[:,1:,:].detach()
        scalar_before = torch.nn.functional.pad(scalar_before,(0,0,1,0))
        scalar_after = torch.nn.functional.pad(scalar_after,(0,0,0,1))

        mask = (scalar_importance.lt(scalar_before)) & (scalar_importance.lt(scalar_after))
        mask = mask.reshape(scalar_importance.shape[0], -1)
        mask[:,0] = True
        batch_index = mask.nonzero()[:,0]
        valley_index_start = mask.nonzero()[:,1]
        mask[:,0] = False
        mask[:,-1] = True
        valley_index_end = mask.nonzero()[:,1] + 2
        valley_index_end = torch.where(valley_index_end > (length) * torch.ones_like(valley_index_end), 
                                       (length) * torch.ones_like(valley_index_end), valley_index_end)

        _,counts = torch.unique(batch_index, return_counts = True)
        # print("counts: ", counts)
        max_counts = (torch.max(counts)).item()

        utri_mat1 = torch.tril(torch.ones(max_counts+1,max_counts),-1).to(xs_pad.device)
        batch_index_mask = utri_mat1[counts]
        batch_index_mask = batch_index_mask.reshape(-1,1)
        batch_index_mask = batch_index_mask.nonzero()[:, 0]

        valleys = torch.zeros(batch * max_counts, 2).type_as(valley_index_start)
        valleys[batch_index_mask] = torch.cat((valley_index_start.unsqueeze(1), valley_index_end.unsqueeze(1)),1)
        # print(valleys)
        
        # utri_mat = torch.tril(torch.cuda.FloatTensor(length+1,length).fill_(1),-1)
        utri_mat = torch.tril(torch.ones(length+1,length),-1).to(xs_pad.device)
        output_mask = (utri_mat[valleys[:,1]]-utri_mat[valleys[:,0]]).reshape(batch, max_counts, length)
        output_mask = output_mask.detach()

        xs_pad = torch.bmm(output_mask, alpha_h) / torch.bmm(output_mask, scalar_importance).clamp_(1e-6)
        # print(torch.isnan(output).any())
        
        # olens = (olens / olens[0] * xs_pad.shape[1]).type_as(olens)
        olens = counts
        
        return xs_pad, olens
