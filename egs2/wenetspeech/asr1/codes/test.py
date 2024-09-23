'''
Author: FnoY fangying@westlake.edu.cn
LastEditors: FnoY0723 fangying@westlake.edu.cn
LastEditTime: 2024-09-05 15:21:58
FilePath: /espnet/egs2/realman/codes/test.py
'''
import os
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument('--test_path', type=str, default=None, help='Path to the test data')
args = parser.parse_args()

# test_path = "/data/home/fangying/sn_enh_mel/mels/4xSPB_Hid96_offline_L1_1e-5_prevdata"
test_name = args.test_path.split('/')[-1]
json_path = f"/data/home/fangying/espnet/egs2/realman/results/{test_name}.json"

os.system(f"python /data/home/fangying/espnet/egs2/realman/codes/0_inference.py --json_file {json_path}")
print("Inference Done!\n")

os.system(f"python /data/home/fangying/espnet/egs2/realman/codes/1_convert_json_to_trn.py --json_file {json_path}")
print("Conversion Done!\n")

os.system(f"python /data/home/fangying/espnet/egs2/realman/codes/3_calculate_cer.py --transcript_name {test_name}")
print("CER Calculation Done!\n")

time.sleep(1)
os.system(f"/data/home/fangying/anaconda3/envs/espnet/bin/python /data/home/fangying/espnet/egs2/realman/codes/4_summarize_results.py")
print("Results Summarization Done!\n")