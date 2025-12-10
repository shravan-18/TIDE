#!/bin/bash
# Improved automation script for running all ablation studies for DGMHRN
# Usage: ./run_ablations.sh <dataset_dir>

# Check if dataset directory is provided
if [ -z "$1" ]; then
    echo "Usage: ./run_ablations.sh <dataset_dir>"
    exit 1
fi

DATASET_DIR=$1
OUTPUT_DIR="./SUIME_ablation_results"
EPOCHS=32
IMG_SIZE=128
CROP_SIZE=128
BATCH_SIZE=8
WORKERS=4
BASE_LR=0.00001
REFINEMENT_LR=0.000005

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Start time tracking
START_TIME=$(date "+%Y-%m-%d %H:%M:%S")
echo "Starting ablation studies at ${START_TIME}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Dataset directory: ${DATASET_DIR}"
echo ""

# Function to run an ablation study
run_ablation() {
    ABLATION_TYPE=$1
    echo "========================================================"
    echo "Running ablation study: ${ABLATION_TYPE}"
    echo "========================================================"
    
    ABLATION_DIR="${OUTPUT_DIR}/${ABLATION_TYPE}"
    mkdir -p "${ABLATION_DIR}"
    
    python main.py --mode ablation --ablation_type "${ABLATION_TYPE}" \
      --data_dir "${DATASET_DIR}" --output_dir "${OUTPUT_DIR}" \
      --num_epochs ${EPOCHS} --img_size ${IMG_SIZE} --crop_size ${CROP_SIZE} \
      --batch_size ${BATCH_SIZE} --num_workers ${WORKERS} --lr ${BASE_LR} \
      --refinement_lr ${REFINEMENT_LR} --log_dir "${ABLATION_DIR}/logs" \
      --save_dir "${ABLATION_DIR}/checkpoints" --mixed_precision \
      --grad_clip 0.1 --weight_decay 0.0001 --visualize --use_improved_ablation
    
    if [ $? -ne 0 ]; then
        echo "Error running ablation study: ${ABLATION_TYPE}"
        echo "Continuing with next ablation study..."
    else
        echo "Successfully completed ablation study: ${ABLATION_TYPE}"
    fi
    echo ""
}

# Run all ablation studies
run_ablation "no_degradation_maps"
run_ablation "single_hypothesis"
run_ablation "fusion_type"
run_ablation "no_diversity_loss"
run_ablation "decoder_types"
run_ablation "no_refinement"
run_ablation "refinement_magnitude"

# Calculate total runtime
END_TIME=$(date "+%Y-%m-%d %H:%M:%S")
echo "========================================================"
echo "All ablation studies completed!"
echo "Started at: ${START_TIME}"
echo "Ended at: ${END_TIME}"
echo "Results saved in: ${OUTPUT_DIR}"
echo "========================================================"

# Generate combined report
python -c "
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load all results
results = {}
ablation_types = ['no_degradation_maps', 'single_hypothesis', 'fusion_type', 
                 'no_diversity_loss', 'decoder_types', 'no_refinement', 'refinement_magnitude']

for ablation_type in ablation_types:
    result_path = os.path.join('${OUTPUT_DIR}', ablation_type, 'all_results.csv')
    if os.path.exists(result_path):
        try:
            df = pd.read_csv(result_path, index_col=0)
            results[ablation_type] = df
        except Exception as e:
            print(f'Error loading {ablation_type} results: {e}')

# Create summary report
with open(os.path.join('${OUTPUT_DIR}', 'summary_report.txt'), 'w') as f:
    f.write('DGMHRN Ablation Studies Summary Report\\n')
    f.write('=======================================\\n\\n')
    
    for ablation_type, df in results.items():
        f.write(f'{ablation_type.replace(\"_\", \" \").title()} Ablation\\n')
        f.write('-' * 40 + '\\n')
        f.write(df.to_string() + '\\n\\n')
    
    f.write('\\nPSNR Comparison Across Ablations\\n')
    f.write('-' * 40 + '\\n')
    
    # Get best PSNR for each ablation type
    best_psnr = {}
    for ablation_type, df in results.items():
        if 'psnr' in df.columns:
            try:
                best_variant = df['psnr'].idxmax()
                best_psnr[ablation_type] = {
                    'variant': best_variant,
                    'psnr': df.loc[best_variant, 'psnr']
                }
            except Exception as e:
                print(f'Error finding best PSNR for {ablation_type}: {e}')
    
    # Sort by PSNR
    sorted_best = sorted(best_psnr.items(), key=lambda x: x[1]['psnr'], reverse=True)
    
    for ablation_type, info in sorted_best:
        f.write(f'{ablation_type}: {info[\"variant\"]} - PSNR: {info[\"psnr\"]:.4f}\\n')

    # Create visualization if we have results
    if results:
        try:
            plt.figure(figsize=(12, 8))
            
            # Collect PSNR values for each ablation
            ablation_names = []
            psnr_values = []
            
            for ablation_type, info in sorted_best:
                ablation_names.append(ablation_type.replace('_', ' ').title())
                psnr_values.append(info['psnr'])
            
            # Create bar chart
            bars = plt.bar(ablation_names, psnr_values)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{height:.2f}', ha='center', va='bottom')
            
            plt.title('PSNR Comparison Across Ablations')
            plt.ylabel('PSNR (dB)')
            plt.ylim(0, max(psnr_values) * 1.2)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            # Save figure
            plt.savefig(os.path.join('${OUTPUT_DIR}', 'psnr_comparison.png'), dpi=300)
            plt.close()
            
            print('Generated PSNR comparison chart')
            
            # Also create a detailed chart with variants for each ablation
            plt.figure(figsize=(15, 10))
            
            # Collect data for each ablation and its variants
            current_pos = 0
            tick_positions = []
            tick_labels = []
            
            # Colors for different ablation types
            colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
            
            for i, (ablation_type, df) in enumerate(results.items()):
                if 'psnr' in df.columns:
                    try:
                        # Get variants for this ablation
                        variants = df.index.tolist()
                        psnr_values = df['psnr'].tolist()
                        
                        # Create bar chart for this ablation
                        width = 0.8
                        positions = [current_pos + j for j in range(len(variants))]
                        plt.bar(positions, psnr_values, width=width, label=ablation_type, color=colors[i])
                        
                        # Add value labels
                        for pos, val in zip(positions, psnr_values):
                            plt.text(pos, val + 0.1, f'{val:.2f}', ha='center', va='bottom', rotation=90)
                        
                        # Save positions for ticks
                        tick_positions.extend(positions)
                        tick_labels.extend(variants)
                        
                        # Update current position
                        current_pos += len(variants) + 1  # Add space between ablation types
                    except Exception as e:
                        print(f'Error creating detailed chart for {ablation_type}: {e}')
            
            plt.xticks(tick_positions, tick_labels, rotation=90)
            plt.legend(title='Ablation Type')
            plt.title('PSNR for All Ablation Variants')
            plt.ylabel('PSNR (dB)')
            plt.tight_layout()
            
            # Save figure
            plt.savefig(os.path.join('${OUTPUT_DIR}', 'detailed_psnr_comparison.png'), dpi=300)
            plt.close()
            
            print('Generated detailed PSNR comparison chart')
            
        except Exception as e:
            print(f'Error creating visualizations: {e}')

print('Summary report generated: ${OUTPUT_DIR}/summary_report.txt')
"

echo "Done."
