#!/bin/bash


# ==================================================
# TimeMMD NewYork AQI
#
# Text Weight Fine-grained Search
#
# Fixed:
#   prompt_weight = 0.5
#   text_mode = real
#
# Search:
#   text_weight
#
# Purpose:
#   Find optimal contribution of text residual
#
# ==================================================


set -e



# ==================================================
# Experiment logging
# ==================================================

LOG_ROOT="./experiment_logs/text_weight_search_$(date +%Y%m%d_%H%M)"

mkdir -p ${LOG_ROOT}


echo ""
echo "======================================"
echo "Text Weight Search"
echo "Logs:"
echo ${LOG_ROOT}
echo "======================================"



# ==================================================
# Run one experiment
# ==================================================

run_exp(){

WEIGHT=$1


NAME="NY_AQI_MM_BERT_tw${WEIGHT}"



echo ""
echo "======================================"
echo "Running ${NAME}"
echo "text_weight=${WEIGHT}"
echo "prompt_weight=0.5"
echo "text_mode=real"
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
--text_weight ${WEIGHT} \
--text_mode real



# ==================================================
# Save results
# ==================================================

SAVE_DIR=${LOG_ROOT}/${NAME}

mkdir -p ${SAVE_DIR}



# metric result

cp result_longterm_forecast \
${SAVE_DIR}/result.txt



# checkpoint

mkdir -p ${SAVE_DIR}/checkpoints

cp -r checkpoints/* \
${SAVE_DIR}/checkpoints/



# prediction

mkdir -p ${SAVE_DIR}/results

cp -r results/* \
${SAVE_DIR}/results/



echo "${NAME} finished"

}



# ==================================================
# Search range
# ==================================================


run_exp 0.00

run_exp 0.05

run_exp 0.10

run_exp 0.15

run_exp 0.20

run_exp 0.25

run_exp 0.30

run_exp 0.35

run_exp 0.40

run_exp 0.50



echo ""
echo "======================================"
echo "Text weight search finished"
echo "Results saved at:"
echo ${LOG_ROOT}
echo "======================================"