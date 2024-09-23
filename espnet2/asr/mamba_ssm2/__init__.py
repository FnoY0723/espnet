'''
Author: FnoY fangying@westlake.edu.cn
LastEditors: FnoY0723 fangying@westlake.edu.cn
LastEditTime: 2024-07-18 13:47:20
FilePath: /espnet/espnet2/asr/mamba_ssm2/__init__.py
'''
__version__ = "2.2.2"

from espnet2.asr.mamba_ssm2.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn
from espnet2.asr.mamba_ssm2.modules.mamba_simple import Mamba
from espnet2.asr.mamba_ssm2.modules.mamba2 import Mamba2
from espnet2.asr.mamba_ssm2.models.mixer_seq_simple import MambaLMHeadModel
