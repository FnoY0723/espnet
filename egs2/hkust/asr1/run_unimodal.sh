#!/usr/bin/env bash
###
 # @Author: FnoY 1084585914@qq.com
 # @Date: 2023-04-28 17:33:51
 # @LastEditors: FnoY 1084585914@qq.com
 # @LastEditTime: 2023-06-29 13:33:26
 # @FilePath: /espnet/egs2/hkust/asr1/run_unimodal.sh
 # @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
### 
###
 # @Author: FnoY 1084585914@qq.com
 # @Date: 2023-04-28 17:33:51
 # @LastEditors: FnoY 1084585914@qq.com
 # @LastEditTime: 2023-06-27 09:52:39
 # @FilePath: /espnet/egs2/hkust/asr1/run_unimodal.sh
 # @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
### 
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
CUDA_VISIBLE_DEVICES=4
set -e
set -u
set -o pipefail

train_set=train_nodup
valid_set=train_dev
test_sets="dev"


asr_config=conf/train_asr_uma_transformer.yaml
inference_config=conf/decode_uma.yaml

lm_config=conf/tuning/train_lm_transformer.yaml
use_lm=false
expdir=exp_umatransformer_18e_58
inference_asr_model=valid.cer.ave_10best.pth

# speed perturbation related
# (train_set will be "${train_set}_sp" if speed_perturb_factors is specified)
speed_perturb_factors="0.9 1.0 1.1"

./asr_unimodal.sh                                               \
    --nj 64 \
    --inference_nj 1  \
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