#!/bin/bash


# ==================================================
# TimeMMD NewYork AQI
#
# Text Weight Ablation
#
# Study:
# How much does text embedding contribute?
#
# prompt_weight fixed = 0.5
#
# text_weight:
#
# 0.0  no text
# 0.25 weak text
# 0.5 original
# 0.75 strong text
# 1.0 full text
#
# ==================================================


set -e


LOG_ROOT="./experiment_logs/text_weight_ablation_$(date +%Y%m%d_%H%M)"

mkdir -p ${LOG_ROOT}


echo "Experiment logs:"
echo ${LOG_ROOT}



run_exp(){

TEXT_WEIGHT=$1


NAME="NY_AQI_MM_BERT_tw${TEXT_WEIGHT}"



echo ""
echo "======================================"
echo "Running ${NAME}"
echo "text_weight=${TEXT_WEIGHT}"
echo "======================================"



echo "GPU status:"
nvidia-smi --query-gpu=name,memory.used --format=csv



python run.py \
--task_name long_term_forecast \
--is_training 1 \
--model_id ${NAME} \
--model PatchTST \
--data custom \
--root_path ./data/TimeMMD/Environment \
--data_path NewYork_AQI_Day.csv \
--target OT \
--seq_len 96 \
--label_len 48 \
--pred_len 96 \
--batch_size 32 \
--train_epochs 10 \
--llm_model BERT \
--text_len 4 \
--prompt_weight 0.5 \
--text_weight ${TEXT_WEIGHT}



SAVE_DIR=${LOG_ROOT}/${NAME}


mkdir -p ${SAVE_DIR}



cp result_longterm_forecast \
${SAVE_DIR}/result.txt



mkdir -p ${SAVE_DIR}/checkpoints

cp -r checkpoints/* \
${SAVE_DIR}/checkpoints/



mkdir -p ${SAVE_DIR}/results

cp -r results/* \
${SAVE_DIR}/results/



echo "${NAME} finished"



}



# ==================================================
# experiments
# ==================================================


run_exp 0.0

run_exp 0.25

run_exp 0.5

run_exp 0.75

run_exp 1.0



echo ""

echo "======================================"
echo "All text weight ablation experiments done"
echo "Results saved at:"
echo ${LOG_ROOT}
echo "======================================"