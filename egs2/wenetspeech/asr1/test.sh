#!/usr/bin/env bash
###
 # @Author: FnoY fangying@westlake.edu.cn
 # @LastEditors: FnoY0723 fangying@westlake.edu.cn
 # @LastEditTime: 2024-10-08 19:15:50
 # @FilePath: /espnet/egs2/wenetspeech/asr1/test.sh
### 
enhPath="/data/home/fangying/sn_enh_mel"
expPath="/data/home/fangying/espnet/egs2/wenetspeech/asr1/exp_ctc_44_32_60/asr_train_asr_raw_zh_char/decode_asr_asr_model_valid.acc.ave_10best/test_meeting"
parentDir="$(dirname "$expPath")"
file="/data/home/fangying/espnet/espnet2/asr/frontend/default.py"
declare -a dirArray
mel_path=$1
dirArray+=("$mel_path")
# dirArray+=("/data/home/fangying/sn_enh_mel/mels/8xSPB_Hid128_offline_real_rts_ensemble139-149")



for i in "${dirArray[@]}"; do
    sed -i "108c \ \ \ \ \ \ \ \ self.base_mels_path = \"$i\"" "$file"
    echo "$i"
    ./run.sh --stage 12 --stop_stage 13
    newName=$(basename "$i")
    newPath="${parentDir}/${newName}"
    echo "$newName"
    mv "$expPath" "$newPath"
done 

./run.sh --stage 13 --stop_stage 13
