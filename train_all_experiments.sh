#!/bin/bash

DATASET_NAME=$1
DATA_DIR=data/
BASE_CKPT=checkpoints/best_model.pth
REFINE_CKPT=checkpoints/refinement/best_refinement_model.pth
RESULTS_DIR=results/${DATASET_NAME}/ablations

echo "Running ablation studies for dataset: ${DATASET_NAME}"
mkdir -p ${RESULTS_DIR} 2>/dev/null

echo "Running Base Model Ablations..."

echo "Running No Degradation Maps ablation..."
python main.py --mode ablation --data_dir ${DATA_DIR} --checkpoint ${BASE_CKPT} --ablation_type no_degradation_maps --output_dir ${RESULTS_DIR}/no_degradation_maps --visualize || echo "Failed: No Degradation Maps ablation but continuing..."

echo "Running Single Hypothesis ablation..."
python main.py --mode ablation --data_dir ${DATA_DIR} --checkpoint ${BASE_CKPT} --ablation_type single_hypothesis --output_dir ${RESULTS_DIR}/single_hypothesis --visualize || echo "Failed: Single Hypothesis ablation but continuing..."

echo "Running Fusion Types ablation..."
python main.py --mode ablation --data_dir ${DATA_DIR} --checkpoint ${BASE_CKPT} --ablation_type fusion_type --output_dir ${RESULTS_DIR}/fusion_types --visualize || echo "Failed: Fusion Types ablation but continuing..."

echo "Running No Diversity Loss ablation..."
python main.py --mode ablation --data_dir ${DATA_DIR} --checkpoint ${BASE_CKPT} --ablation_type no_diversity_loss --output_dir ${RESULTS_DIR}/no_diversity_loss --visualize || echo "Failed: No Diversity Loss ablation but continuing..."

echo "Running Decoder Types ablation..."
python main.py --mode ablation --data_dir ${DATA_DIR} --checkpoint ${BASE_CKPT} --ablation_type decoder_types --output_dir ${RESULTS_DIR}/decoder_types --visualize || echo "Failed: Decoder Types ablation but continuing..."

echo "Running Progressive Refinement Ablations..."

echo "Running Refinement Magnitude ablation..."
python main.py --mode ablation --data_dir ${DATA_DIR} --progressive_checkpoint ${REFINE_CKPT} --ablation_type refinement_magnitude --output_dir ${RESULTS_DIR}/refinement_magnitude --visualize || echo "Failed: Refinement Magnitude ablation but continuing..."

echo "Running No Refinement ablation..."
python main.py --mode ablation --data_dir ${DATA_DIR} --progressive_checkpoint ${REFINE_CKPT} --ablation_type no_refinement --output_dir ${RESULTS_DIR}/no_refinement --visualize || echo "Failed: No Refinement ablation but continuing..."

echo "All ablation studies completed for dataset: ${DATASET_NAME}"
