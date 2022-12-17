#!/bin/bash

set -ex

curdir=$(pwd)
echo "curdir:$curdir"
cd "$curdir" || exit

cd ../../


base_dir="/mnt/cephfs/hjh/train_record/vc/freevc/dataset/vctk"

wav_dir="${base_dir}/wav48"
sr1=16000
sr2=22050
out_dir1="${base_dir}/wav16k"
out_dir2="${base_dir}/wav22050"


#--------------------------
# downsample
#--------------------------
#python downsample.py --in_dir ${wav_dir} --sr1 ${sr1} --sr2 ${sr2} --out_dir1 ${out_dir1} --out_dir2 ${out_dir2}


#--------------------------
# train meta file
#--------------------------

train_meta_file="${base_dir}/train_16k.txt"
dev_meta_file="${base_dir}/dev_16k.txt"
test_meta_file="${base_dir}/test_16k.txt"

#python preprocess_flist.py --train_list ${train_meta_file} --val_list ${dev_meta_file} --test_list ${test_meta_file} --source_dir ${out_dir1}

#--------------------------
# spk embedding
#--------------------------

out_dir_root="${base_dir}/embed_16k"
spk_encoder_ckpt="`pwd`/speaker_encoder/ckpt/pretrained_bak_5805000.pt"
#CUDA_VISIBLE_DEVICES=0 python preprocess_spk.py --in_dir ${out_dir1} --num_workers 12 --out_dir_root ${out_dir_root} --spk_encoder_ckpt ${spk_encoder_ckpt}


#--------------------------
# ssl
#--------------------------
ssl_dir="${base_dir}/ssl_16k"
wavlm_model="/mnt/cephfs/hjh/train_record/vc/freevc/pretrain_models/WavLM-Large.pt"
#CUDA_VISIBLE_DEVICES=0 python preprocess_ssl.py --sr ${sr1} --in_dir ${out_dir1} --out_dir ${ssl_dir} --wavlm_model ${wavlm_model}


#--------------------------
# sr
#--------------------------

hifigan_config_file="/mnt/cephfs/hjh/train_record/vc/freevc/pretrain_models/config.json"
hifigan_ckpt="/mnt/cephfs/hjh/train_record/vc/freevc/pretrain_models/generator_v1"
save_wav_dir="${base_dir}/sr/wav16k"
ssl_dir="${base_dir}/sr/wavlm16k"
CUDA_VISIBLE_DEVICES=1 python -u preprocess_sr.py --min 68 --max 72 --sr ${sr1} --config ${hifigan_config_file} --hifigan_ckpt ${hifigan_ckpt} --wavlm_model ${wavlm_model} --in_dir ${out_dir1} --wav_dir "${save_wav_dir}_v1" --ssl_dir "${ssl_dir}_v1"||true &
CUDA_VISIBLE_DEVICES=1 python -u preprocess_sr.py --min 73 --max 76 --sr ${sr1} --config ${hifigan_config_file} --hifigan_ckpt ${hifigan_ckpt} --wavlm_model ${wavlm_model} --in_dir ${out_dir1} --wav_dir "${save_wav_dir}_v2" --ssl_dir "${ssl_dir}_v2"||true &
CUDA_VISIBLE_DEVICES=2 python -u preprocess_sr.py --min 77 --max 80 --sr ${sr1} --config ${hifigan_config_file} --hifigan_ckpt ${hifigan_ckpt} --wavlm_model ${wavlm_model} --in_dir ${out_dir1} --wav_dir "${save_wav_dir}_v3" --ssl_dir "${ssl_dir}_v3"||true &
CUDA_VISIBLE_DEVICES=2 python -u preprocess_sr.py --min 81 --max 84 --sr ${sr1} --config ${hifigan_config_file} --hifigan_ckpt ${hifigan_ckpt} --wavlm_model ${wavlm_model} --in_dir ${out_dir1} --wav_dir "${save_wav_dir}_v4" --ssl_dir "${ssl_dir}_v4"||true &
CUDA_VISIBLE_DEVICES=3 python -u preprocess_sr.py --min 85 --max 88 --sr ${sr1} --config ${hifigan_config_file} --hifigan_ckpt ${hifigan_ckpt} --wavlm_model ${wavlm_model} --in_dir ${out_dir1} --wav_dir "${save_wav_dir}_v5" --ssl_dir "${ssl_dir}_v5"||true &
CUDA_VISIBLE_DEVICES=3 python -u preprocess_sr.py --min 89 --max 92 --sr ${sr1} --config ${hifigan_config_file} --hifigan_ckpt ${hifigan_ckpt} --wavlm_model ${wavlm_model} --in_dir ${out_dir1} --wav_dir "${save_wav_dir}_v6" --ssl_dir "${ssl_dir}_v6"||true &