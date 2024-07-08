'''
Author: FnoY fangying@westlake.edu.cn
LastEditors: FnoY0723 fangying@westlake.edu.cn
LastEditTime: 2024-07-03 15:12:31
FilePath: /espnet/espnet2/layers/test_mel.py
'''
from typing import Tuple
import numpy as np
import pandas as pd
import librosa
import torch

from espnet.nets.pytorch_backend.nets_utils import make_pad_mask

_mel_options = dict(
    sr=16000,
    n_fft=512,
    n_mels=80,
    fmin=0.0,
    fmax=8000,
    htk=False,
)

melmat = librosa.filters.mel(**_mel_options)
pd.DataFrame(melmat).to_csv("/data/home/fangying/espnet/espnet2/layers/splaney_melmat.csv", index=False)
print(melmat)