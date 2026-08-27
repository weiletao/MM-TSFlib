#!/bin/bash

# ==================================================
# TimeMMD NewYork AQI
#
# Adaptive Text Gate Ablation
#
# Fixed:
#   prompt_weight = 0.5
#   adaptive text gate enabled
#
# Modes:
#   real
#   zero
#   shuffle
#   random
#
# ==================================================

set -e


# ==================================================
# Experiment logging
# ==================================================

LOG_ROOT="./experiment_logs/text_gate_ablation_$(date +%Y%m%d_%H%M)"

mkdir -p ${LOG_ROOT}

echo "Experiment logs:"
echo ${LOG_ROOT}



# ==================================================
# Run experiment
# ==================================================

run_exp(){

MODE=$1

NAME="NY_AQI_MM_BERT_TextGate_text_${MODE}"


echo ""
echo "======================================"
echo "Running ${NAME}"
echo "text_mode=${MODE}"
echo "prompt_weight=0.5"
echo "adaptive text gate enabled"
echo "======================================"


echo "GPU status:"
nvidia-smi --query-gpu=name,memory.used --format=csv



# ==================================================
# Clean previous generated files
# ==================================================

rm -rf checkpoints/*
rm -rf results/*
rm -rf test_results/*
rm -f result_longterm_forecast



# ==================================================
# Run training
# ==================================================

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
--text_mode ${MODE}



# ==================================================
# Save outputs
# ==================================================

SAVE_DIR=${LOG_ROOT}/${NAME}

mkdir -p ${SAVE_DIR}


echo "Saving results to ${SAVE_DIR}"



# metrics

if [ -f result_longterm_forecast ]; then

cp result_longterm_forecast \
${SAVE_DIR}/result.txt

fi



# checkpoints

mkdir -p ${SAVE_DIR}/checkpoints

if [ "$(ls -A checkpoints 2>/dev/null)" ]; then

cp -r checkpoints/* \
${SAVE_DIR}/checkpoints/

fi



# prediction results

mkdir -p ${SAVE_DIR}/results

if [ "$(ls -A results 2>/dev/null)" ]; then

cp -r results/* \
${SAVE_DIR}/results/

fi



echo "${NAME} finished"

}



# ==================================================
# Run ablation
# ==================================================

run_exp real

run_exp zero

run_exp shuffle

run_exp random



echo ""

echo "======================================"
echo "All adaptive text gate ablation experiments done"
echo "Results saved at:"
echo ${LOG_ROOT}
echo "======================================"