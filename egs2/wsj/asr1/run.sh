#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
CUDA_VISIBLE_DEVICES="7"
set -e
set -u
set -o pipefail

train_set=train_si284
valid_set=test_dev93
test_sets="test_dev93 test_eval92"

asr_config=conf/train_asr_unimodal_conformer_12e.yaml
inference_config=conf/decode_asr_unimodal_attention.yaml
use_lm=false
lm_config=conf/train_lm_transformer.yaml
# use_wordlm=false
expdir=exp_umaconformer_12e_noenc_426
inference_asr_model=valid.cer.ave_10best.pth

./asr.sh \
    --nj 64 \
    --inference_nj 64 \
    --ngpu 1 \
    --lang en \
    --use_lm true \
    --token_type word \
    --use_lm ${use_lm}                                 \
    --expdir ${expdir}                                 \
    --inference_asr_model ${inference_asr_model}       \
    --lm_config "${lm_config}"                         \
    --asr_config "${asr_config}"                       \
    --inference_config "${inference_config}"           \
    --nbpe 80 \
    --nlsyms_txt data/nlsyms.txt \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --bpe_train_text "data/train_si284/text" \
    --lm_train_text "data/train_si284/text data/local/other_text/text" "$@"
