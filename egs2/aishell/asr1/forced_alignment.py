'''
Author: FnoY fangying@westlake.edu.cn
LastEditors: FnoY0723 fangying@westlake.edu.cn
LastEditTime: 2024-07-09 18:57:28
FilePath: /espnet/egs2/aishell/asr1/forced_alignment.py
'''
import os
os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '1'
os.environ["OMP_NUM_THREADS"] = str(4)

test_path = '/data/home/fangying/espnet/egs2/aishell/asr1/downloads/data_aishell/wav/test'
for speaker in os.listdir(test_path):
    # os.system('mfa validate /data/home/fangying/espnet/egs2/aishell/asr1/downloads/data_aishell/wav/test/S0764 mandarin_mfa mandarin_mfa --single_speaker -j 8')
    os.system(f'mfa align {test_path}/{speaker} mandarin_mfa mandarin_mfa /data/home/fangying/espnet/egs2/aishell/asr1/downloads/forced_alignment/test/{speaker} --clean --verbose --single_speaker -j 8')