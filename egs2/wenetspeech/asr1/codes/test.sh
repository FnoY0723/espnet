#!/usr/bin/env bash
###
 # @Author: FnoY fangying@westlake.edu.cn
 # @LastEditors: FnoY0723 fangying@westlake.edu.cn
 # @LastEditTime: 2024-09-11 15:39:11
 # @FilePath: /espnet/egs2/realman/codes/test.sh
### 

file="/data/home/fangying/espnet/espnet2/asr/frontend/default.py"
declare -a dirArray
# dirArray+=("/data/home/fangying/sn_enh_mel/mels/4xFSB_Hid96_offline_L1_1e-5_ensemble89-99")
# dirArray+=("/data/home/fangying/sn_enh_mel/mels/4xFSB_Hid96_offline_MSE_1e-5_ensemble89-99")
dirArray+=("/data/home/fangying/sn_enh_mel/mels/4xSPB_Hid96_offline_L1_1e-5_EN+CN_ensemble89-99")
# dirArray+=("/data/home/fangying/sn_enh_mel/mels/8xSPB_Hid128_offline_ensemble89-99")
# dirArray+=("/data/home/fangying/sn_enh_mel/mels/4xSPB_Hid96_offline_L1_1e-5_prevdata")
# dirArray+=("/data/home/fangying/sn_enh_mel/mels/4xSPB_Hid96_fixnorm_offline_ensemble89-99")

# for d in $(ls -d "$enhPath"/*/); do
#     dirArray+=("$d")
# done

for i in "${dirArray[@]}"; do
    sed -i "108c \ \ \ \ \ \ \ \ self.base_mels_path = \"$i\"" "$file"
    echo "$i"
    python /data/home/fangying/espnet/egs2/realman/codes/test.py --test_path "$i"
done



