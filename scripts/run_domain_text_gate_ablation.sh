#!/bin/bash

# ==================================================
# TimeMMD Full-Domain Ablation
#
# Adaptive Text Gate  vs  Fixed text_weight
#
# Arms:
#   gate   : --use_text_gate 1            (text_weight ignored)
#   fixed  : --use_text_gate 0 --text_weight 0.20
#
# Shared protocol (same as NewYork_AQI gate ablation):
#   PatchTST / BERT / sl96 / label48 / pl96 / bs32 / epochs10
#   prompt_weight 0.5 / text_mode real
#
# Cleanup policy:
#   - before EACH experiment: remove residue of previous runs
#     (checkpoints/ results/ test_results/ result_longterm_forecast)
#   - after ALL experiments succeed (every result.txt archived):
#     remove this session's residue as well
#
# ==================================================

set -e


# ==================================================
# Domain registry (add Natural when its CSV is available)
# ==================================================

DOMAINS=(
"Algriculture"
"Climate"
"Economy"
"Energy"
"Environment"
"Public_Health"
"Security"
"SocialGood"
"Traffic"
#"Natural"
)

CSV_PATHS=(
"./data/TimeMMD/Algriculture/US_RetailBroilerComposite_Month.csv"
"./data/TimeMMD/Climate/US_precipitation_month.csv"
"./data/TimeMMD/Economy/US_TradeBalance_Month.csv"
"./data/TimeMMD/Energy/US_GasolinePrice_Week.csv"
"./data/TimeMMD/Environment/NewYork_AQI_Day.csv"
"./data/TimeMMD/Public_Health/US_FLURATIO_Week.csv"
"./data/TimeMMD/Security/US_FEMAGrant_Month.csv"
"./data/TimeMMD/SocialGood/Unadj_UnemploymentRate_ALL_processed.csv"
"./data/TimeMMD/Traffic/US_VMT_Month.csv"
#"./data/TimeMMD/Natural/<Natural_CSV>.csv"
)

# Per-domain windows aligned with DOMAINS order:
#   monthly series        -> seq 24 / label 12 / pred 12
#   weekly-daily series   -> seq 96 / label 48 / pred 96
# val-split safety: n_window(val) = round(0.1*N) - pred + 1 must stay > 0

SEQ_LENS=(
24
24
24
96
96
96
24
24
24
)

LABEL_LENS=(
12
12
12
48
48
48
12
12
12
)

PRED_LENS=(
12
12
12
96
96
96
12
12
12
)


# ==================================================
# Experiment logging
# ==================================================

LOG_ROOT="./experiment_logs/domain_text_gate_ablation_$(date +%Y%m%d_%H%M)"

mkdir -p ${LOG_ROOT}

echo "Experiment logs:"
echo ${LOG_ROOT}


# Track every archived run dir for the final success check

SAVED_DIRS=()


run_exp(){

DOM_INDEX=$1
ARM=$2

DOMAIN=${DOMAINS[${DOM_INDEX}]}
CSV_PATH=${CSV_PATHS[${DOM_INDEX}]}
SEQ_LEN=${SEQ_LENS[${DOM_INDEX}]}
LABEL_LEN=${LABEL_LENS[${DOM_INDEX}]}
PRED_LEN=${PRED_LENS[${DOM_INDEX}]}

if [ "${ARM}" = "gate" ]; then
NAME="${DOMAIN}_MM_BERT_TextGate"
GATE_ARGS="--use_text_gate 1"
else
NAME="${DOMAIN}_MM_BERT_tw0.20"
GATE_ARGS="--use_text_gate 0 --text_weight 0.20"
fi


echo ""
echo "======================================"
echo "Running ${NAME}"
echo "domain=${DOMAIN}"
echo "csv=${CSV_PATH}"
echo "windows: seq=${SEQ_LEN} label=${LABEL_LEN} pred=${PRED_LEN}"
echo "arm=${ARM} (${GATE_ARGS})"
echo "======================================"


echo "GPU status:"
nvidia-smi --query-gpu=name,memory.used --format=csv || true


# ==================================================
# Clean residue left by previous experiment(s)
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
--root_path $(dirname ${CSV_PATH}) \
--data_path $(basename ${CSV_PATH}) \
--target OT \
--seq_len ${SEQ_LEN} \
--label_len ${LABEL_LEN} \
--pred_len ${PRED_LEN} \
--batch_size 32 \
--train_epochs 10 \
--llm_model BERT \
--text_len 4 \
--prompt_weight 0.5 \
--text_mode real \
${GATE_ARGS}



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


# prediction results (includes gate.npy and text_gate_hist.pdf)

mkdir -p ${SAVE_DIR}/results

if [ "$(ls -A results 2>/dev/null)" ]; then

cp -r results/* \
${SAVE_DIR}/results/

fi


SAVED_DIRS+=("${SAVE_DIR}")

echo "${NAME} finished"

}



# ==================================================
# Run ablation: 9 domains x 2 arms
# ==================================================

for i in "${!DOMAINS[@]}"; do

run_exp ${i} fixed

run_exp ${i} gate

done



# ==================================================
# Verify all runs produced logs, then clean this session's residue
# ==================================================

ALL_OK=1

for d in "${SAVED_DIRS[@]}"; do

if [ ! -f "${d}/result.txt" ]; then
echo "WARNING: missing result.txt in ${d}"
ALL_OK=0
fi

done


if [ "${ALL_OK}" = "1" ]; then

echo ""
echo "All ${#SAVED_DIRS[@]} runs archived successfully."

echo "Cleaning this session's residue ..."

rm -rf checkpoints/*
rm -rf results/*
rm -rf test_results/*
rm -f result_longterm_forecast

else

echo ""
echo "Some runs failed to archive result.txt."
echo "Residue kept for inspection under checkpoints/ results/ test_results/"

fi



echo ""
echo "======================================"
echo "Full-domain text gate ablation done"
echo "Results saved at:"
echo ${LOG_ROOT}
echo "======================================"