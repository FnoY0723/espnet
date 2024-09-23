#!/usr/bin/env bash
###
 # @Author: FnoY fangying@westlake.edu.cn
 # @LastEditors: FnoY0723 fangying@westlake.edu.cn
 # @LastEditTime: 2024-09-11 16:53:11
 # @FilePath: /espnet/egs2/wenetspeech/asr1/run.sh
### 
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

set=L    # S for the small set, M for the mediate set, L for the large set

train_set=train_"$(echo "${set}" | tr "[:lower:]" "[:upper:]")"
valid_set=dev
test_sets="test_meeting"
# test_sets="dev test_meeting test_net"
# test_sets="hkust_dev"

asr_config=conf/train_asr.yaml
inference_config=conf/decode_asr.yaml

lm_config=conf/train_lm.yaml
use_lm=false

# speed perturbation related
# add "--speed_perturb_factors="0.9 1.0 1.1" if you want to
# apply speed perturbation for the training data

./asr.sh                                               \
    --lang zh                                          \
    --local_data_opts "--set ${set}"                   \
    --audio_format flac                                 \
    --feats_type raw                                   \
    --token_type char                                  \
    --use_lm ${use_lm}                                 \
    --lm_config "${lm_config}"                         \
    --asr_config "${asr_config}"                       \
    --inference_config "${inference_config}"           \
    --train_set "${train_set}"                         \
    --valid_set "${valid_set}"                         \
    --test_sets "${test_sets}"                         \
    --asr_speech_fold_length 512 \
    --asr_text_fold_length 150 \
    --lm_fold_length 150 \
    --lm_train_text "data/${train_set}/text" "$@"
