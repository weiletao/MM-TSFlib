#!/bin/bash

# ==================================================
# TimeMMD NewYork AQI
# Prompt Weight Ablation
#
# prompt_weight interpolation:
#
# 0.0  : only time series branch
# 1.0  : text-guided prediction branch
#
# ==================================================

set -e


# ==================================================
# Experiment logging
# ==================================================

LOG_ROOT="./experiment_logs/prompt_weight_ablation_$(date +%Y%m%d_%H%M)"

mkdir -p ${LOG_ROOT}


echo "Experiment logs:"
echo ${LOG_ROOT}


# ==================================================
# Run one experiment
# ==================================================

run_exp(){

NAME=$1
PW=$2


echo ""
echo "======================================"
echo "Running ${NAME}"
echo "prompt_weight=${PW}"
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
--prompt_weight ${PW}



# save experiment outputs

SAVE_DIR=${LOG_ROOT}/${NAME}

mkdir -p ${SAVE_DIR}


echo "Saving results to ${SAVE_DIR}"


# save metrics

cp result_longterm_forecast \
${SAVE_DIR}/result.txt


# save corresponding checkpoints

mkdir -p ${SAVE_DIR}/checkpoints

cp -r checkpoints/* \
${SAVE_DIR}/checkpoints/


# save prediction results

mkdir -p ${SAVE_DIR}/results

cp -r results/* \
${SAVE_DIR}/results/


echo "${NAME} finished"

}



# ==================================================
# Modality interpolation experiments
# ==================================================

run_exp NY_AQI_MM_BERT_pw000 0.00

run_exp NY_AQI_MM_BERT_pw001 0.01

run_exp NY_AQI_MM_BERT_pw005 0.05

run_exp NY_AQI_MM_BERT_pw010 0.10

run_exp NY_AQI_MM_BERT_pw020 0.20

run_exp NY_AQI_MM_BERT_pw050 0.50

run_exp NY_AQI_MM_BERT_pw080 0.80

run_exp NY_AQI_MM_BERT_pw100 1.00



echo ""

echo "======================================"
echo "All prompt weight experiments done"
echo "Results saved at:"
echo ${LOG_ROOT}
echo "======================================"