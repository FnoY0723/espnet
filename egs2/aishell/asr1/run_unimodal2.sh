#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
CUDA_VISIBLE_DEVICES=3
set -e
set -u
set -o pipefail

train_set=train
valid_set=dev
test_sets="test"

asr_config=conf/train_asr_unimodal_conformer.yaml
inference_config=conf/decode_asr_unimodal_attention.yaml

lm_config=conf/train_lm_transformer.yaml
use_lm=false
use_wordlm=false
expdir=exp_umaconformer_421_enc
inference_asr_model=valid.cer.ave_10best.pth

# speed perturbation related
# (train_set will be "${train_set}_sp" if speed_perturb_factors is specified)
speed_perturb_factors="0.9 1.0 1.1"

./asr_unimodal.sh \
    --nj 64 \
    --inference_nj 64 \
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
