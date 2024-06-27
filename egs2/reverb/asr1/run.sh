#!/usr/bin/env bash
###
 # @Author: FnoY fangying@westlake.edu.cn
 # @LastEditors: FnoY0723 fangying@westlake.edu.cn
 # @LastEditTime: 2024-06-26 21:18:43
 # @FilePath: /espnet/egs2/reverb/asr1/run.sh
### 
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

# REVERB Official dataset
# train_set=tr_simu_8ch_si284
# WSJ0 + WSJ1 + WSJ cam0 (Clean speech only)
train_set=tr_wsjcam0_si284
valid_set=dt_mult_1ch
test_sets="et_real_1ch"

./asr.sh \
    --lang "en" \
    --feats_normalize utterance_mvn \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --use_lm true \
    --token_type char \
    --nbpe 80 \
    --audio_format flac \
    --nlsyms_txt data/nlsyms.txt \
    --inference_config conf/tuning/decode.yaml \
    --lm_config conf/tuning/train_lm_transformer.yaml \
    --asr_config conf/tuning/train_asr_transformer4.yaml \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --inference_asr_model valid.acc.ave_10best.pth \
    --lm_train_text "${train_set}/text data/local/other_text/text" "$@"
