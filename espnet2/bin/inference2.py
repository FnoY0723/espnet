'''
Author: FnoY fangying@westlake.edu.cn
LastEditors: FnoY0723 fangying@westlake.edu.cn
LastEditTime: 2024-05-18 12:54:04
FilePath: /espnet/espnet2/bin/inference2.py
'''
import os
os.environ["OMP_NUM_THREADS"] = str(4) # 限制进程数量，放在import torch和numpy之前。不加会导致程序占用特别多的CPU资源，使得服务器变卡。 
# limit the threads to reduce cpu overloads, will speed up when there are lots of CPU cores on the running machine 
lang = 'zh'
fs = 16000 
# tag = 'espnet/pengcheng_guo_wenetspeech_asr_train_asr_raw_zh_char'

import time
# import torch

import resampy
# import pandas as pd
import soundfile

import json
import tqdm
from typing import List
import multiprocessing as mp
import numpy as np

aligned_results = '/data/home/RealisticAudio/codes/4_target_gen/aligned_static_high_精细对齐_correctDPRIR_mcnn_v16epoch219_filtered'
aligned_name = aligned_results.split('/')[-1]
json_file = f'/data/home/RealisticAudio/codes/4_target_gen/asr_results/wenetspeech_asr_model/{aligned_name}.json'

def procss(files):
    
    def text_normalizer(text):
        text = text.upper()
        return text.translate(str.maketrans('', '', string.punctuation))

    try:
        from espnet2.bin.asr_inference import Speech2Text
        import string

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
        for index, file in enumerate(files):            
            wav_path = file
            speech, rate = soundfile.read(wav_path)
            ma = np.max(np.abs(speech))
            speech = speech / ma
            speech = resampy.resample(speech, rate, fs, axis=0)
            nbests = speech2text(speech)
            text, *_ = nbests[0]
            json_obj = json.dumps([wav_path, text_normalizer(text)], ensure_ascii=False)
            excep=True
            while excep:
                try :
                    file_output = open(json_file, 'a', encoding='utf-8')
                    excep=False
                except:
                    time.sleep(1)
                
            file_output.write(json_obj + ',\n')
            file_output.close()

            print(f"{wav_path}: {text_normalizer(text)}")
            print("*" * 50)
    except Exception as e:
        print(f"{e}")

def init_env_var(gpus: List[int]):
    i = gpus.get()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(i)
    print(os.getpid(),os.environ['CUDA_VISIBLE_DEVICES'])


if __name__ == '__main__':
    gpus = [0,1,2,3,4,5,6,7]
    
    mp.set_start_method('spawn')

    queue = mp.Queue()
    for gid in gpus:
        queue.put(gid)

    all_files = []
    for root, dirs, files in os.walk(aligned_results):
        for index, file in enumerate(files):
            if file.endswith('.wav'):
                all_files.append(os.path.join(root, file))

    files = all_files
    files.sort()

    pbar = tqdm.tqdm(total=len(files))
    pbar.set_description('generating rirs') 
    
    p = mp.Pool(processes=len(gpus), initializer=init_env_var, initargs=(queue,))    
    filess=[files[:len(files)//8], files[len(files)//8:len(files)//4], files[len(files)//4:len(files)//4+len(files)//8], 
             files[len(files)//4+len(files)//8:len(files)//2], files[len(files)//2:len(files)//2+len(files)//8], files[len(files)//2+len(files)//8:len(files)//2+len(files)//4], 
             files[len(files)//2+len(files)//4: len(files)//2+len(files)//4+len(files)//8], files[len(files)//2+len(files)//4+len(files)//8:]]
    
    p.map(procss, filess)
    p.close()
    p.join()