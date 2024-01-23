#!/usr/bin/env bash
###
 # @Author: FnoY fangying@westlake.edu.cn
 # @LastEditors: FnoY 1084585914@qq.com
 # @LastEditTime: 2024-01-22 21:01:10
 # @FilePath: /espnet/egs2/hkust/asr1/run.sh
### 
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train_nodup
valid_set=train_dev
test_sets="dev"


asr_config=conf/train_asr_transformer.yaml
inference_config=conf/decode.yaml

lm_config=conf/tuning/train_lm_transformer.yaml
use_lm=false
expdir=exp_transformer_53
# inference_asr_model=valid.acc.ave_10best.pth
# inference_asr_model=valid.acc.ave_10best.pth
inference_asr_model=valid.cer_ctc.ave_10best.pth
# speed perturbation related
# (train_set will be "${train_set}_sp" if speed_perturb_factors is specified)
speed_perturb_factors="0.9 1.0 1.1"

./asr.sh                                               \
    --nj 64 \
    --inference_nj 64  \
    --ngpu 1 \
    --lang zh                                          \
    --audio_format flac                                \
    --feats_type raw                                   \
    --token_type char                                  \
    --nlsyms_txt data/nlsyms.txt \
    --use_lm ${use_lm}                                 \
    --expdir ${expdir}                                 \
    --inference_asr_model ${inference_asr_model}       \
    --lm_config "${lm_config}"                         \
    --asr_config "${asr_config}"                       \
    --inference_config "${inference_config}"           \
    --train_set "${train_set}"                         \
    --valid_set "${valid_set}"                         \
    --test_sets "${test_sets}"                         \
    --speed_perturb_factors "${speed_perturb_factors}" \
    --lm_train_text "data/${train_set}/text" "$@"
