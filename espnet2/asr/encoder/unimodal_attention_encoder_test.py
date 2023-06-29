# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Unimodal attention encoder definition."""

from typing import List, Optional, Tuple

import torch
from typeguard import check_argument_types

from espnet2.asr.ctc import CTC
from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding
from espnet.nets.pytorch_backend.transformer.encoder_layer import EncoderLayer
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
# from espnet2.asr.uma import UMA

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
        # Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]

        # masks = (~make_pad_mask(olens)[:, None, :]).to(xs_pad.device)
        # unimodal attention
        # plt.clf()
        # plt.plot(xs_pad.cpu().detach().numpy()[0,:,1], color='black')
        batch, length, _ = xs_pad.size()

        scalar_importance = self.linear_sigmoid(xs_pad)
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
        
        # return xs_pad, olens, scalar_importance
        return xs_pad, olens


class UnimodalAttentionEncoder(AbsEncoder):
    """Transformer encoder module.

    Args:
        input_size: input dim
        output_size: dimension of attention
        attention_heads: the number of heads of multi head attention
        linear_units: the number of units of position-wise feed forward
        num_blocks: the number of decoder blocks
        dropout_rate: dropout rate
        attention_dropout_rate: dropout rate in attention
        positional_dropout_rate: dropout rate after adding positional encoding
        input_layer: input layer type
        pos_enc_class: PositionalEncoding or ScaledPositionalEncoding
        normalize_before: whether to use layer_norm before the first block
        concat_after: whether to concat attention layer's input and output
            if True, additional linear will be applied.
            i.e. x -> x + linear(concat(x, att(x)))
            if False, no additional linear will be applied.
            i.e. x -> x + att(x)
        positionwise_layer_type: linear of conv1d
        positionwise_conv_kernel_size: kernel size of positionwise conv1d layer
        padding_idx: padding_idx for input_layer=embed
    """

    def __init__(
        self,
        input_size: int,
        output_size: int = 256,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        input_layer: Optional[str] = "conv2d",
        pos_enc_class=PositionalEncoding,
        normalize_before: bool = True,
        concat_after: bool = False,
        positionwise_layer_type: str = "linear",
        positionwise_conv_kernel_size: int = 1,
        padding_idx: int = -1,
        interctc_layer_idx: List[int] = [],
        interctc_use_conditioning: bool = False,
    ):
        assert check_argument_types()
        super().__init__()
        self._output_size = output_size

        if input_layer == "linear":
            self.embed = torch.nn.Sequential(
                torch.nn.Linear(input_size, output_size),
                torch.nn.LayerNorm(output_size),
                torch.nn.Dropout(dropout_rate),
                torch.nn.ReLU(),
                pos_enc_class(output_size, positional_dropout_rate),
            )
        elif input_layer == "conv2d":
            self.embed = Conv2dSubsampling(input_size, output_size, dropout_rate)
        elif input_layer == "conv2d1":
            self.embed = Conv2dSubsampling1(input_size, output_size, dropout_rate)
        elif input_layer == "conv2d2":
            self.embed = Conv2dSubsampling2(input_size, output_size, dropout_rate)
        elif input_layer == "conv2d6":
            self.embed = Conv2dSubsampling6(input_size, output_size, dropout_rate)
        elif input_layer == "conv2d8":
            self.embed = Conv2dSubsampling8(input_size, output_size, dropout_rate)
        elif input_layer == "embed":
            self.embed = torch.nn.Sequential(
                torch.nn.Embedding(input_size, output_size, padding_idx=padding_idx),
                pos_enc_class(output_size, positional_dropout_rate),
            )
        elif input_layer is None:
            if input_size == output_size:
                self.embed = None
            else:
                self.embed = torch.nn.Linear(input_size, output_size)
        else:
            raise ValueError("unknown input_layer: " + input_layer)
        self.normalize_before = normalize_before
        if positionwise_layer_type == "linear":
            positionwise_layer = PositionwiseFeedForward
            positionwise_layer_args = (
                output_size,
                linear_units,
                dropout_rate,
            )
        elif positionwise_layer_type == "conv1d":
            positionwise_layer = MultiLayeredConv1d
            positionwise_layer_args = (
                output_size,
                linear_units,
                positionwise_conv_kernel_size,
                dropout_rate,
            )
        elif positionwise_layer_type == "conv1d-linear":
            positionwise_layer = Conv1dLinear
            positionwise_layer_args = (
                output_size,
                linear_units,
                positionwise_conv_kernel_size,
                dropout_rate,
            )
        else:
            raise NotImplementedError("Support only linear or conv1d.")
        self.encoders = repeat(
            num_blocks,
            lambda lnum: EncoderLayer(
                output_size,
                MultiHeadedAttention(
                    attention_heads, output_size, attention_dropout_rate
                ),
                positionwise_layer(*positionwise_layer_args),
                dropout_rate,
                normalize_before,
                concat_after,
            ),
        )
        if self.normalize_before:
            self.after_norm = LayerNorm(output_size)

        self.interctc_layer_idx = interctc_layer_idx
        if len(interctc_layer_idx) > 0:
            assert 0 < min(interctc_layer_idx) and max(interctc_layer_idx) < num_blocks
        self.interctc_use_conditioning = interctc_use_conditioning
        self.conditioning_layer = None

        self.uma = UMA(input_size, output_size)
        # self.linear_sigmoid = torch.nn.Sequential(
        #     torch.nn.Linear(256, 1),
        #     torch.nn.Sigmoid(),
        # )

    def output_size(self) -> int:
        return self._output_size

    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        prev_states: torch.Tensor = None,
        ctc: CTC = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Embed positions in tensor.

        Args:
            xs_pad: input tensor (B, L, D)
            ilens: input length (B)
            prev_states: Not to be used now.
        Returns:
            position embedded tensor and mask
        """
        masks = (~make_pad_mask(ilens)[:, None, :]).to(xs_pad.device)

        if self.embed is None:
            xs_pad = xs_pad
        elif (
            isinstance(self.embed, Conv2dSubsampling)
            or isinstance(self.embed, Conv2dSubsampling1)
            or isinstance(self.embed, Conv2dSubsampling2)
            or isinstance(self.embed, Conv2dSubsampling6)
            or isinstance(self.embed, Conv2dSubsampling8)
        ):
            short_status, limit_size = check_short_utt(self.embed, xs_pad.size(1))
            if short_status:
                raise TooShortUttError(
                    f"has {xs_pad.size(1)} frames and is too short for subsampling "
                    + f"(it needs more than {limit_size} frames), return empty results",
                    xs_pad.size(1),
                    limit_size,
                )
            xs_pad, masks = self.embed(xs_pad, masks)
        else:
            xs_pad = self.embed(xs_pad)

        xs_pad, masks = self.encoders(xs_pad, masks)

        if self.normalize_before:
            xs_pad = self.after_norm(xs_pad)
    
        olens = masks.squeeze(1).sum(1)

        xs_pad, olens = self.uma(xs_pad, olens)

        # ################################################################################################
        # # unimodal attention
        # batch, length, _ = xs_pad.size()

        # scalar_importance = self.linear_sigmoid(xs_pad)
        # # scalar_importance = torch.nn.functional.log_softmax(scalar_importance)
        # # print("alpha: ", scalar_importance)

        # alpha_h = torch.mul(scalar_importance, xs_pad)

        # # find valleys' index
        # scalar_before = scalar_importance[:,:-1,:].detach()
        # scalar_after = scalar_importance[:,1:,:].detach()
        # scalar_before = torch.nn.functional.pad(scalar_before,(0,0,1,0))
        # scalar_after = torch.nn.functional.pad(scalar_after,(0,0,0,1))

        # mask = (scalar_importance.lt(scalar_before)) & (scalar_importance.lt(scalar_after))
        # mask = mask.reshape(scalar_importance.shape[0], -1)
        # mask[:,0] = True
        # batch_index = mask.nonzero()[:,0]
        # valley_index_start = mask.nonzero()[:,1]
        # mask[:,0] = False
        # mask[:,-1] = True
        # valley_index_end = mask.nonzero()[:,1] + 2
        # valley_index_end = torch.where(valley_index_end > (length) * torch.ones_like(valley_index_end), 
        #                                (length) * torch.ones_like(valley_index_end), valley_index_end)

        # _,counts = torch.unique(batch_index, return_counts = True)
        # max_counts = (torch.max(counts)).item()

        # utri_mat1 = torch.tril(torch.ones(max_counts+1,max_counts),-1).to(xs_pad.device)
        # batch_index_mask = utri_mat1[counts]
        # batch_index_mask = batch_index_mask.reshape(-1,1)
        # batch_index_mask = batch_index_mask.nonzero()[:, 0]

        # valleys = torch.zeros(batch * max_counts, 2).type_as(valley_index_start)
        # valleys[batch_index_mask] = torch.cat((valley_index_start.unsqueeze(1), valley_index_end.unsqueeze(1)),1)
        # # print(valleys)
        
        # # utri_mat = torch.tril(torch.cuda.FloatTensor(length+1,length).fill_(1),-1)
        # utri_mat = torch.tril(torch.ones(length+1,length),-1).to(xs_pad.device)
        # output_mask = (utri_mat[valleys[:,1]]-utri_mat[valleys[:,0]]).reshape(batch, max_counts, length)
        # output_mask = output_mask.detach()

        # xs_pad = torch.bmm(output_mask, alpha_h) / torch.bmm(output_mask, scalar_importance).clamp_(1e-6)
        # # print(torch.isnan(output).any())
        
        # # olens = (olens / olens[0] * xs_pad.shape[1]).type_as(olens)
        # olens = counts
        
        return xs_pad, olens, None
    
def setup_seed(seed):
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    torch.random.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__=="__main__":
    setup_seed(11)
    uma = UnimodalAttentionEncoder(256,256)
    xs_pad = torch.randn(32,10,256)
    ilens = torch.randint(1,10,(32,))
    xs, olens, _ = uma(xs_pad, ilens)
    print(xs)