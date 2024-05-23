'''
Author: FnoY fangying@westlake.edu.cn
LastEditors: FnoY0723 fangying@westlake.edu.cn
LastEditTime: 2024-05-16 16:53:48
FilePath: /espnet/espnet2/bin/inference.py
'''
import os
os.environ["OMP_NUM_THREADS"] = str(16) # 限制进程数量，放在import torch和numpy之前。不加会导致程序占用特别多的CPU资源，使得服务器变卡。 
# limit the threads to reduce cpu overloads, will speed up when there are lots of CPU cores on the running machine 
lang = 'zh'
fs = 16000 
# tag = 'espnet/pengcheng_guo_wenetspeech_asr_train_asr_raw_zh_char'

import time
import torch
import string
import resampy
from espnet2.bin.asr_inference import Speech2Text
import pandas as pd
import soundfile

import json

speech2text = Speech2Text(
    asr_train_config='/data/home/fangying/espnet/egs2/wenetspeech/asr1/exp_ctc_44_32_60/espnet/pengcheng_guo_wenetspeech_asr_train_asr_raw_zh_char/config.yaml',
    asr_model_file='/data/home/fangying/espnet/egs2/wenetspeech/asr1/exp_ctc_44_32_60/espnet/pengcheng_guo_wenetspeech_asr_train_asr_raw_zh_char/valid.acc.ave_10best.pth',
    device="cpu",
    minlenratio=0.0,
    maxlenratio=0.0,
    ctc_weight=0.6,
    beam_size=10,
    batch_size=0,
    nbest=1
)

def text_normalizer(text):
    text = text.upper()
    return text.translate(str.maketrans('', '', string.punctuation))

json_file = '/data/home/RealisticAudio/codes/4_target_gen/asr_results/wenetspeech_asr_model/aligned_static_low_精细对齐_correctDPRIR_filtered3_downto8k_fy.json'
scenes = ['scene_0305_LabOffice', 'scene_0306_A2park', 'scene_0307_c18two', 'scene_0308_badminton_court', 
          'scene_0311_basketball', 'scene_0409_canteen', 'scene_0411_1号门大厅']

with open(json_file, 'a', encoding='utf-8') as file_output:
    file_output.write('[\n')
    for scene in scenes:
        for root, dirs, files in os.walk(f'/data/home/RealisticAudio/codes/4_target_gen/aligned_static_low_精细对齐_correctDPRIR_filtered3/{scene}'):
            for index, file in enumerate(files):
                if file.endswith('.wav'):
                    wav_path = os.path.join(root, file)
                    speech, rate = soundfile.read(wav_path)
                    speech = resampy.resample(speech, rate, 8000, axis=0)
                    speech = resampy.resample(speech, 8000, fs, axis=0)
                    nbests = speech2text(speech)
                    text, *_ = nbests[0]

                    json_obj = json.dumps([wav_path, text_normalizer(text)], ensure_ascii=False)
                    file_output.write(json_obj + ',\n')

                    print(f"{wav_path}: {text_normalizer(text)}")
                    print("*" * 50)

    file_output.write(']\n')
    