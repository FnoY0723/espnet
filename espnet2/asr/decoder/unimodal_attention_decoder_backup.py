# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Decoder definition."""
from typing import Any, List, Sequence, Tuple

import torch
from typeguard import check_argument_types

from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from espnet.nets.pytorch_backend.transformer.encoder_layer import EncoderLayer
from espnet.nets.pytorch_backend.transformer.decoder_layer import DecoderLayer
from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import (
    PositionwiseFeedForward,
)
from espnet.nets.pytorch_backend.transformer.repeat import repeat


class BaseTransformerDecoder(AbsDecoder):
    """Base class of Transfomer decoder module.

    Args:
        vocab_size: output dim
        encoder_output_size: dimension of attention
        attention_heads: the number of heads of multi head attention
        linear_units: the number of units of position-wise feed forward
        num_blocks: the number of decoder blocks
        dropout_rate: dropout rate
        self_attention_dropout_rate: dropout rate for attention
        input_layer: input layer type
        use_output_layer: whether to use output layer
        pos_enc_class: PositionalEncoding or ScaledPositionalEncoding
        normalize_before: whether to use layer_norm before the first block
        concat_after: whether to concat attention layer's input and output
            if True, additional linear will be applied.
            i.e. x -> x + linear(concat(x, att(x)))
            if False, no additional linear will be applied.
            i.e. x -> x + att(x)
    """

    def __init__(
        self,
        vocab_size: int,
        encoder_output_size: int,
        output_size: int = 256,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        use_output_layer: bool = False,
        pos_enc_class=PositionalEncoding,
        normalize_before: bool = True,
    ):
        assert check_argument_types()
        super().__init__()
        attention_dim = encoder_output_size
        self._output_size = output_size

        # self.linear_sigmoid = torch.nn.Sequential(
        #     torch.nn.Linear(256, 1),
        #     torch.nn.Sigmoid(),
        # )
        
        self.embed = torch.nn.Sequential(
            torch.nn.Linear(attention_dim, output_size),
            torch.nn.LayerNorm(output_size),
            torch.nn.Dropout(dropout_rate),
            torch.nn.ReLU(),
            pos_enc_class(output_size, positional_dropout_rate),
        )

        self.normalize_before = normalize_before
        if self.normalize_before:
            self.after_norm = LayerNorm(output_size)
        if use_output_layer:
            self.output_layer = torch.nn.Linear(output_size, output_size)
        else:
            self.output_layer = None

        # Must set by the inheritance
        self.decoders = None
    
    def output_size(self) -> int:
        return self._output_size

    def forward(
        self,
        hs_pad: torch.Tensor,
        hlens: torch.Tensor,
        ys_in_pad: torch.Tensor,
        ys_in_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward decoder.

        Args:
            hs_pad: encoded memory, float32  (batch, maxlen_in, feat)
            hlens: (batch)
            ys_in_pad:
                input token ids, int64 (batch, maxlen_out)
                if input_layer == "embed"
                input tensor (batch, maxlen_out, #mels) in the other cases
            ys_in_lens: (batch)
        Returns:
            (tuple): tuple containing:

            x: decoded token score before softmax (batch, maxlen_out, token)
                if use_output_layer is True,
            olens: (batch, )
        """
        # ###################### UMA #######################
        # batch, length, _ = hs_pad.size()
        # scalar_importance = self.linear_sigmoid(hs_pad)
        
        # # plt.plot(scalar_importance.cpu().detach().numpy()[0,:,:], color='blue')
        # # plt.savefig(f'./inference_images/hyp_{hlens[0]}.png')
        # # k = k+1
        
        # # score = self.pooling_proj(hs_pad) / self._output_size**0.5 # (batch, T, 1)
        # # scalar_importance = torch.softmax(score, dim=1)

        # # print("alpha: ", scalar_importance)
        # alpha_h = torch.mul(scalar_importance, hs_pad)

        # # find valleys' index
        # scalar_before = scalar_importance[:,:-1,:].detach()
        # scalar_after = scalar_importance[:,1:,:].detach()
        # scalar_before = torch.nn.functional.pad(scalar_before,(0,0,1,0))
        # scalar_after = torch.nn.functional.pad(scalar_after,(0,0,0,1))

        # mask = ((scalar_importance.lt(scalar_before)) & (scalar_importance.lt(scalar_after)))
        # mask = mask.reshape(scalar_importance.shape[0], -1)
        # # print(mask.shape)
        # mask[:,0] = True
        # batch_index = mask.nonzero()[:,0]
        # valley_index_start = mask.nonzero()[:,1]
        # mask[:,0] = False
        # mask[:,-1] = True
        # valley_index_end = valley_index_start + 2
        # valley_index_end = torch.where(valley_index_end > (length) * torch.ones_like(valley_index_end), 
        #                                (length) * torch.ones_like(valley_index_end), valley_index_end)
        # # print(valley_index_start.shape)
        # # print(valley_index_end.shape)

        # _,counts = torch.unique(batch_index, return_counts = True)
        # # logging.info(str(counts))
        # max_counts = (torch.max(counts)).item()

        # utri_mat1 = torch.tril(torch.ones(max_counts+1,max_counts),-1).to(hs_pad.device)
        # batch_index_mask = utri_mat1[counts]
        # batch_index_mask = batch_index_mask.reshape(-1,1)
        # batch_index_mask = batch_index_mask.nonzero()[:, 0]

        # valleys = torch.zeros(batch * max_counts, 2).type_as(valley_index_start)
        # valleys[batch_index_mask] = torch.cat((valley_index_start.unsqueeze(1), valley_index_end.unsqueeze(1)),1)
        # # logging.info(str(valleys))
        
        # # utri_mat = torch.tril(torch.cuda.FloatTensor(length+1,length).fill_(1),-1)
        # utri_mat = torch.tril(torch.ones(length+1,length),-1).to(hs_pad.device)
        # output_mask = (utri_mat[valleys[:,1]]-utri_mat[valleys[:,0]]).reshape(batch, max_counts, length)
        # output_mask = output_mask.detach()

        # hs_pad = torch.bmm(output_mask, alpha_h) / torch.bmm(output_mask, scalar_importance).clamp_(1e-6)
        # # print(torch.isnan(output).any())
        
        # hlens = (hlens / hlens[0] * hs_pad.shape[1]).type_as(hlens)
        #  # hlens = counts
        # #######################################################################################################
        masks = (~make_pad_mask(hlens)[:, None, :]).to(hs_pad.device)
        
        hs_pad = self.embed(hs_pad)

        hs_pad, masks = self.decoders(hs_pad, masks)

        if self.normalize_before:
            hs_pad = self.after_norm(hs_pad)
        if self.output_layer is not None:
            hs_pad = self.output_layer(hs_pad)

        olens = masks.squeeze(1).sum(1)
        return hs_pad, olens


class UnimodalAttentionDecoder(BaseTransformerDecoder):
    def __init__(
        self,
        vocab_size: int,
        encoder_output_size: int,
        output_size: int = 256,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        self_attention_dropout_rate: float = 0.0,
        src_attention_dropout_rate: float = 0.0,
        use_output_layer: bool = True,
        pos_enc_class=PositionalEncoding,
        normalize_before: bool = True,
        concat_after: bool = False,
        layer_drop_rate: float = 0.0,
    ):
        assert check_argument_types()
        super().__init__(
            vocab_size=vocab_size,
            encoder_output_size=encoder_output_size,
            output_size=output_size,
            dropout_rate=dropout_rate,
            positional_dropout_rate=positional_dropout_rate,
            use_output_layer=use_output_layer,
            pos_enc_class=pos_enc_class,
            normalize_before=normalize_before,
        )

        self.decoders = repeat(
            num_blocks,
            lambda lnum: EncoderLayer(
                output_size,
                MultiHeadedAttention(
                    attention_heads, output_size, self_attention_dropout_rate
                ),
                PositionwiseFeedForward(output_size, linear_units, dropout_rate),
                dropout_rate,
                normalize_before,
                concat_after,
            ),
        )
