#!/bin/bash

# ==================================================
# TimeMMD NewYork AQI
#
# Weighted Text Modality Ablation
#
# Fixed:
#   prompt_weight = 0.5
#   text_weight   = 0.20
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

LOG_ROOT="./experiment_logs/text_weighted_ablation_$(date +%Y%m%d_%H%M)"

mkdir -p ${LOG_ROOT}

echo "Experiment logs:"
echo ${LOG_ROOT}



# ==================================================
# Run experiment
# ==================================================

run_exp(){

MODE=$1

NAME="NY_AQI_MM_BERT_tw0.20_text_${MODE}"


echo ""
echo "======================================"
echo "Running ${NAME}"
echo "text_mode=${MODE}"
echo "prompt_weight=0.5"
echo "text_weight=0.20"
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
--text_weight 0.20 \
--text_mode ${MODE}



# ==================================================
# Save outputs
# ==================================================

SAVE_DIR=${LOG_ROOT}/${NAME}

mkdir -p ${SAVE_DIR}


echo "Saving results to ${SAVE_DIR}"


# metrics
cp result_longterm_forecast \
${SAVE_DIR}/result.txt



# checkpoints
mkdir -p ${SAVE_DIR}/checkpoints

cp -r checkpoints/* \
${SAVE_DIR}/checkpoints/



# prediction results
mkdir -p ${SAVE_DIR}/results

cp -r results/* \
${SAVE_DIR}/results/



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
echo "All weighted text ablation experiments done"
echo "Results saved at:"
echo ${LOG_ROOT}
echo "======================================"