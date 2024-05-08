#!/usr/bin/env bash
###
 # @Author: FnoY fangying@westlake.edu.cn
 # @LastEditors: FnoY0723 fangying@westlake.edu.cn
 # @LastEditTime: 2024-04-29 11:14:45
 # @FilePath: /espnet/egs2/aishell/asr1/run_ctc.sh
### 
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train
valid_set=dev
test_sets="test"

asr_config=conf/train_asr_transformer_ctc.yaml
inference_config=conf/decode_asr_ctc.yaml

lm_config=conf/train_lm_transformer.yaml
use_lm=false
use_wordlm=false
expdir=exp_ctc_412_64_60
inference_asr_model=valid.cer_ctc.ave_10best.pth
# inference_asr_model=valid.acc.ave_10best.pth

# speed perturbation related
# (train_set will be "${train_set}_sp" if speed_perturb_factors is specified)
speed_perturb_factors="0.9 1.0 1.1"

./asr.sh \
    --nj 32 \
    --inference_nj 32 \
    --ngpu 1 \
    --lang zh \
    --audio_format "flac.ark" \
    --feats_type raw \
    --token_type char \
    --use_lm ${use_lm}                                 \
    --use_word_lm ${use_wordlm}                        \
    --expdir ${expdir}                                 \
    --inference_asr_model ${inference_asr_model}       \
    --lm_config "${lm_config}"                         \
    --asr_config "${asr_config}"                       \
    --inference_config "${inference_config}"           \
    --train_set "${train_set}"                         \
    --valid_set "${valid_set}"                         \
    --test_sets "${test_sets}"                         \
    --speed_perturb_factors "${speed_perturb_factors}" \
    --asr_speech_fold_length 512 \
    --asr_text_fold_length 150 \
    --lm_fold_length 150 \
    --lm_train_text "data/${train_set}/text" "$@"
