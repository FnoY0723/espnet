#!/usr/bin/env bash
###
 # @Author: FnoY fangying@westlake.edu.cn
 # @LastEditors: FnoY0723 fangying@westlake.edu.cn
 # @LastEditTime: 2024-07-05 16:26:29
 # @FilePath: /espnet/egs2/chime4/asr1/run.sh
### 
###
 # @Author: FnoY fangying@westlake.edu.cn
 # @LastEditors: FnoY 1084585914@qq.com
 # @LastEditTime: 2023-11-07 13:13:37
 # @FilePath: /espnet/egs2/chime4/asr1/run.sh
### 
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
CUDA_VISIBLE_DEVICES="4,5"
set -e
set -u
set -o pipefail

train_set=tr05_multi_noisy_si284 # tr05_multi_noisy (original training data) or tr05_multi_noisy_si284 (add si284 data)
valid_set=dt05_multi_isolated_1ch_track
test_sets="\
et05_real_isolated_1ch_track\
"
# dt05_real_beamformit_2mics dt05_simu_beamformit_2mics et05_real_beamformit_2mics et05_simu_beamformit_2mics \
# dt05_real_beamformit_5mics dt05_simu_beamformit_5mics et05_real_beamformit_5mics et05_simu_beamformit_5mics \
asr_config=conf/tuning/train_asr_e_branchformer_e10_mlp1024_linear1024_macaron_lr1e-3_warmup25k.yaml
inference_config=conf/decode_asr.yaml
inference_config=conf/decode_asr_transformer.yaml
lm_config=conf/train_lm_transformer.yaml

speed_perturb_factors="0.9 1.0 1.1"

use_word_lm=false
word_vocab_size=65000
nj=64               # The number of parallel jobs.
inference_nj=64 
expdir=exp_branchformer_utterance_mvn

./asr.sh                                   \
    --nj 64 \
    --inference_nj 64 \
    --gpu_inference false \
    --ngpu 2 \
    --lang en \
    --nlsyms_txt data/nlsyms.txt           \
    --token_type char                      \
    --feats_type raw                       \
    --audio_format flac                \
    --asr_config "${asr_config}"           \
    --inference_config "${inference_config}"     \
    --lm_config "${lm_config}"             \
    --use_word_lm ${use_word_lm}           \
    --word_vocab_size ${word_vocab_size}   \
    --train_set "${train_set}"             \
    --valid_set "${valid_set}"             \
    --test_sets "${test_sets}"             \
    --speed_perturb_factors "${speed_perturb_factors}" \
    --bpe_train_text "data/${train_set}/text" \
    --lm_train_text "data/${train_set}/text data/local/other_text/text" "$@"\
    --expdir "${expdir}"
