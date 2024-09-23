#!/usr/bin/env bash
###
 # @Author: FnoY fangying@westlake.edu.cn
 # @LastEditors: FnoY0723 fangying@westlake.edu.cn
 # @LastEditTime: 2024-09-01 18:26:19
 # @FilePath: /espnet/egs2/hkust/asr1/test.sh
### 
enhPath="/data/home/fangying/sn_enh_mel"
expPath="/data/home/fangying/espnet/egs2/hkust/asr1/exp_transformer_53/espnet/pengcheng_guo_wenetspeech_asr_train_asr_raw_zh_char/decode_asr_model_valid.cer_ctc.ave_10best/dev"
parentDir="$(dirname "$expPath")"
file="/data/home/fangying/espnet/espnet2/asr/frontend/default.py"
declare -a dirArray
dirArray+=("/data/home/fangying/sn_enh_mel/4xSPB_Hid96_offline_L1_1e-6_ensemble89-99")

# for d in $(ls -d "$enhPath"/*/); do
#     dirArray+=("$d")
# done

for i in "${dirArray[@]}"; do
    sed -i "108c \ \ \ \ \ \ \ \ self.base_mels_path = \"$i\"" "$file"
    echo "$i"
    ./run.sh --stage 12 --stop_stage 13 --skip_data_prep true --skip_train true --download_model espnet/pengcheng_guo_wenetspeech_asr_train_asr_raw_zh_char
    newName=$(basename "$i")
    newPath="${parentDir}/${newName}_dev"
    echo "$newName"
    mv "$expPath" "$newPath"
done

./run.sh --stage 13 --stop_stage 13 --skip_data_prep true --skip_train true --download_model espnet/pengcheng_guo_wenetspeech_asr_train_asr_raw_zh_char

# expPath="/data/home/fangying/espnet/egs2/chime4/asr1/exp_branchformer_utterance_mvn/asr_train_asr_e_branchformer_e10_mlp1024_linear1024_macaron_lr1e-3_warmup25k_raw_en_char_sp/decode_asr_lm_lm_train_lm_transformer_en_char_valid.loss.ave_asr_model_valid.acc.ave_10best/et05_simu_isolated_1ch_track"
# parentDir="$(dirname "$expPath")"

# for i in "${dirArray[@]}"; do
#     sed -i "108c \ \ \ \ \ \ \ \ self.base_mels_path = \"$i\"" "$file"
#     echo "$i"
#     ./run.sh --stage 12 --stop_stage 13 --test_sets "et05_simu_isolated_1ch_track"
#     newName=$(basename "$i")
#     newPath="${parentDir}/${newName}_simu"
#     echo "$newName"
#     mv "$expPath" "$newPath"
# done

# ./run.sh --stage 13 --stop_stage 13



# for d in $(ls -d "$enhPath"/*/); do
#     dirArray+=("$d")
# done

# for i in "${dirArray[@]}"; do
#     sed -i "106c \ \ \ \ \ \ \ \ self.base_mels_path = \"$i\"" "$file"
#     echo "$i"
#     ./run.sh --stage 12 --stop_stage 13
#     newName=$(basename "$i")
#     newPath="${parentDir}/${newName}"
#     echo "$newName"
#     mv "$expPath" "$newPath"
# done

# ./run.sh --stage 13 --stop_stage 13