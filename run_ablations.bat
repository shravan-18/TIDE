@echo off
REM Improved automation script for running all ablation studies for DGMHRN
REM Usage: run_ablations.bat <dataset_dir>

REM Check if dataset directory is provided
if "%~1"=="" (
    echo Usage: run_ablations.bat ^<dataset_dir^>
    exit /b 1
)

set DATASET_DIR=%~1
set OUTPUT_DIR=.\SUIME_ablation_results
set EPOCHS=32
set IMG_SIZE=128
set CROP_SIZE=128
set BATCH_SIZE=8
set WORKERS=4
set BASE_LR=0.00001
set REFINEMENT_LR=0.000005

REM Create output directory
if not exist %OUTPUT_DIR% mkdir %OUTPUT_DIR%

REM Start time tracking
set start_time=%time%
echo Starting ablation studies at %date% %time%
echo Output directory: %OUTPUT_DIR%
echo Dataset directory: %DATASET_DIR%
echo.

REM Run no_degradation_maps ablation
echo ========================================================
echo Running ablation study: no_degradation_maps
echo ========================================================
set ablation_dir=%OUTPUT_DIR%\no_degradation_maps
if not exist %ablation_dir% mkdir %ablation_dir%
python main.py --mode ablation --ablation_type no_degradation_maps ^
  --data_dir "%DATASET_DIR%" --output_dir "%OUTPUT_DIR%" ^
  --num_epochs %EPOCHS% --img_size %IMG_SIZE% --crop_size %CROP_SIZE% ^
  --batch_size %BATCH_SIZE% --num_workers %WORKERS% --lr %BASE_LR% ^
  --refinement_lr %REFINEMENT_LR% --log_dir "%ablation_dir%\logs" ^
  --save_dir "%ablation_dir%\checkpoints" --mixed_precision ^
  --grad_clip 0.1 --weight_decay 0.0001 --visualize --use_improved_ablation
if %ERRORLEVEL% neq 0 (
    echo Error running ablation study: no_degradation_maps
    echo Continuing with next ablation study...
) else (
    echo Successfully completed ablation study: no_degradation_maps
)
echo.

REM Run single_hypothesis ablation
echo ========================================================
echo Running ablation study: single_hypothesis
echo ========================================================
set ablation_dir=%OUTPUT_DIR%\single_hypothesis
if not exist %ablation_dir% mkdir %ablation_dir%
python main.py --mode ablation --ablation_type single_hypothesis ^
  --data_dir "%DATASET_DIR%" --output_dir "%OUTPUT_DIR%" ^
  --num_epochs %EPOCHS% --img_size %IMG_SIZE% --crop_size %CROP_SIZE% ^
  --batch_size %BATCH_SIZE% --num_workers %WORKERS% --lr %BASE_LR% ^
  --refinement_lr %REFINEMENT_LR% --log_dir "%ablation_dir%\logs" ^
  --save_dir "%ablation_dir%\checkpoints" --mixed_precision ^
  --grad_clip 0.1 --weight_decay 0.0001 --visualize --use_improved_ablation
if %ERRORLEVEL% neq 0 (
    echo Error running ablation study: single_hypothesis
    echo Continuing with next ablation study...
) else (
    echo Successfully completed ablation study: single_hypothesis
)
echo.

REM Run fusion_type ablation
echo ========================================================
echo Running ablation study: fusion_type
echo ========================================================
set ablation_dir=%OUTPUT_DIR%\fusion_type
if not exist %ablation_dir% mkdir %ablation_dir%
python main.py --mode ablation --ablation_type fusion_type ^
  --data_dir "%DATASET_DIR%" --output_dir "%OUTPUT_DIR%" ^
  --num_epochs %EPOCHS% --img_size %IMG_SIZE% --crop_size %CROP_SIZE% ^
  --batch_size %BATCH_SIZE% --num_workers %WORKERS% --lr %BASE_LR% ^
  --refinement_lr %REFINEMENT_LR% --log_dir "%ablation_dir%\logs" ^
  --save_dir "%ablation_dir%\checkpoints" --mixed_precision ^
  --grad_clip 0.1 --weight_decay 0.0001 --visualize --use_improved_ablation
if %ERRORLEVEL% neq 0 (
    echo Error running ablation study: fusion_type
    echo Continuing with next ablation study...
) else (
    echo Successfully completed ablation study: fusion_type
)
echo.

REM Run no_diversity_loss ablation
echo ========================================================
echo Running ablation study: no_diversity_loss
echo ========================================================
set ablation_dir=%OUTPUT_DIR%\no_diversity_loss
if not exist %ablation_dir% mkdir %ablation_dir%
python main.py --mode ablation --ablation_type no_diversity_loss ^
  --data_dir "%DATASET_DIR%" --output_dir "%OUTPUT_DIR%" ^
  --num_epochs %EPOCHS% --img_size %IMG_SIZE% --crop_size %CROP_SIZE% ^
  --batch_size %BATCH_SIZE% --num_workers %WORKERS% --lr %BASE_LR% ^
  --refinement_lr %REFINEMENT_LR% --log_dir "%ablation_dir%\logs" ^
  --save_dir "%ablation_dir%\checkpoints" --mixed_precision ^
  --grad_clip 0.1 --weight_decay 0.0001 --visualize --use_improved_ablation
if %ERRORLEVEL% neq 0 (
    echo Error running ablation study: no_diversity_loss
    echo Continuing with next ablation study...
) else (
    echo Successfully completed ablation study: no_diversity_loss
)
echo.

REM Run decoder_types ablation
echo ========================================================
echo Running ablation study: decoder_types
echo ========================================================
set ablation_dir=%OUTPUT_DIR%\decoder_types
if not exist %ablation_dir% mkdir %ablation_dir%
python main.py --mode ablation --ablation_type decoder_types ^
  --data_dir "%DATASET_DIR%" --output_dir "%OUTPUT_DIR%" ^
  --num_epochs %EPOCHS% --img_size %IMG_SIZE% --crop_size %CROP_SIZE% ^
  --batch_size %BATCH_SIZE% --num_workers %WORKERS% --lr %BASE_LR% ^
  --refinement_lr %REFINEMENT_LR% --log_dir "%ablation_dir%\logs" ^
  --save_dir "%ablation_dir%\checkpoints" --mixed_precision ^
  --grad_clip 0.1 --weight_decay 0.0001 --visualize --use_improved_ablation
if %ERRORLEVEL% neq 0 (
    echo Error running ablation study: decoder_types
    echo Continuing with next ablation study...
) else (
    echo Successfully completed ablation study: decoder_types
)
echo.

REM Run no_refinement ablation
echo ========================================================
echo Running ablation study: no_refinement
echo ========================================================
set ablation_dir=%OUTPUT_DIR%\no_refinement
if not exist %ablation_dir% mkdir %ablation_dir%
python main.py --mode ablation --ablation_type no_refinement ^
  --data_dir "%DATASET_DIR%" --output_dir "%OUTPUT_DIR%" ^
  --num_epochs %EPOCHS% --img_size %IMG_SIZE% --crop_size %CROP_SIZE% ^
  --batch_size %BATCH_SIZE% --num_workers %WORKERS% --lr %BASE_LR% ^
  --refinement_lr %REFINEMENT_LR% --log_dir "%ablation_dir%\logs" ^
  --save_dir "%ablation_dir%\checkpoints" --mixed_precision ^
  --grad_clip 0.1 --weight_decay 0.0001 --visualize --use_improved_ablation
if %ERRORLEVEL% neq 0 (
    echo Error running ablation study: no_refinement
    echo Continuing with next ablation study...
) else (
    echo Successfully completed ablation study: no_refinement
)
echo.

REM Run refinement_magnitude ablation
echo ========================================================
echo Running ablation study: refinement_magnitude
echo ========================================================
set ablation_dir=%OUTPUT_DIR%\refinement_magnitude
if not exist %ablation_dir% mkdir %ablation_dir%
python main.py --mode ablation --ablation_type refinement_magnitude ^
  --data_dir "%DATASET_DIR%" --output_dir "%OUTPUT_DIR%" ^
  --num_epochs %EPOCHS% --img_size %IMG_SIZE% --crop_size %CROP_SIZE% ^
  --batch_size %BATCH_SIZE% --num_workers %WORKERS% --lr %BASE_LR% ^
  --refinement_lr %REFINEMENT_LR% --log_dir "%ablation_dir%\logs" ^
  --save_dir "%ablation_dir%\checkpoints" --mixed_precision ^
  --grad_clip 0.1 --weight_decay 0.0001 --visualize --use_improved_ablation
if %ERRORLEVEL% neq 0 (
    echo Error running ablation study: refinement_magnitude
    echo Continuing with next ablation study...
) else (
    echo Successfully completed ablation study: refinement_magnitude
)
echo.

REM Calculate total runtime
set end_time=%time%
echo ========================================================
echo All ablation studies completed!
echo Started at: %start_time%
echo Ended at: %end_time%
echo Results saved in: %OUTPUT_DIR%
echo ========================================================

REM Generate combined report
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
    result_path = os.path.join('%OUTPUT_DIR%', ablation_type, 'all_results.csv')
    if os.path.exists(result_path):
        try:
            df = pd.read_csv(result_path, index_col=0)
            results[ablation_type] = df
        except Exception as e:
            print(f'Error loading {ablation_type} results: {e}')

# Create summary report
with open(os.path.join('%OUTPUT_DIR%', 'summary_report.txt'), 'w') as f:
    f.write('DGMHRN Ablation Studies Summary Report\n')
    f.write('=======================================\n\n')
    
    for ablation_type, df in results.items():
        f.write(f'{ablation_type.replace(\"_\", \" \").title()} Ablation\n')
        f.write('-' * 40 + '\n')
        f.write(df.to_string() + '\n\n')
    
    f.write('\nPSNR Comparison Across Ablations\n')
    f.write('-' * 40 + '\n')
    
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
        f.write(f'{ablation_type}: {info[\"variant\"]} - PSNR: {info[\"psnr\"]:.4f}\n')

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
            plt.savefig(os.path.join('%OUTPUT_DIR%', 'psnr_comparison.png'), dpi=300)
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
            plt.savefig(os.path.join('%OUTPUT_DIR%', 'detailed_psnr_comparison.png'), dpi=300)
            plt.close()
            
            print('Generated detailed PSNR comparison chart')
            
        except Exception as e:
            print(f'Error creating visualizations: {e}')

print('Summary report generated: %OUTPUT_DIR%/summary_report.txt')
"

echo Done.
