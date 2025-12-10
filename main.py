import argparse
import os
import sys
import torch
import numpy as np
from model import DGMHRN, TIDE
from dataset import get_dataloaders
from utils import count_parameters, load_checkpoint, create_directory, compare_progressive_improvement
from trainer import train_model, train_refinement, finetune_progressive_model, train_progressive_restoration
from eval import evaluate
from ablation_manager import run_improved_ablation_study


def parse_args():
    parser = argparse.ArgumentParser(description='DGMHRN for Underwater Image Restoration with Progressive Refinement Stage')
    parser.add_argument('--mode', type=str, required=True, 
                        choices=['train', 'train_progressive', 'train_refinement', 'finetune', 'eval', 'ablation'],
                        help='Operation mode: train (base model), train_progressive (full pipeline), train_refinement (only refinement stage), finetune (end-to-end), eval, or ablation')
    
    # Common parameters
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    
    # Dataset parameters
    parser.add_argument('--img_size', type=int, default=256, help='Input image size')
    parser.add_argument('--crop_size', type=int, default=256, help='Training crop size')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    
    # Model parameters
    parser.add_argument('--base_channels', type=int, default=64, help='Base number of channels in the model')
    parser.add_argument('--num_downs', type=int, default=5, help='Number of downsampling layers in encoder')
    parser.add_argument('--num_degradation_types', type=int, default=4, help='Number of degradation types (color, contrast, detail, noise) to model')
    parser.add_argument('--norm_type', type=str, default='instance', choices=['batch', 'instance'], help='Normalization type')
    parser.add_argument('--activation', type=str, default='leaky_relu', choices=['relu', 'leaky_relu', 'gelu'], help='Activation function')
    parser.add_argument('--fusion_type', type=str, default='learned', choices=['direct', 'learned', 'attention'], help='Fusion mechanism type')
    
    # Base model training parameters
    parser.add_argument('--num_epochs', type=int, default=300, help='Number of training epochs for base model')
    parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate for base model')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--lr_cycle_epochs', type=int, default=50, help='Cosine annealing cycle length in epochs')
    parser.add_argument('--lr_cycle_mult', type=float, default=1, help='Cosine annealing cycle multiplier')
    parser.add_argument('--warmup_epochs', type=int, default=10, help='Number of warmup epochs')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping threshold')
    parser.add_argument('--mixed_precision', action='store_true', help='Use mixed precision training')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume base training from')
    parser.add_argument('--start_epoch', type=int, default=0, help='Starting epoch number for base training')
    
    # Progressive refinement parameters
    parser.add_argument('--base_model_path', type=str, default=None, help='Path to pre-trained base model checkpoint')
    parser.add_argument('--refinement_epochs', type=int, default=100, help='Number of epochs for refinement training')
    parser.add_argument('--refinement_lr', type=float, default=5e-5, help='Learning rate for refinement training')
    parser.add_argument('--refinement_resume', type=str, default=None, help='Path to resume refinement training from')
    
    # Fine-tuning parameters
    parser.add_argument('--finetune_epochs', type=int, default=50, help='Number of epochs for end-to-end fine-tuning')
    parser.add_argument('--finetune_lr', type=float, default=1e-5, help='Learning rate for fine-tuning')
    parser.add_argument('--finetune_resume', type=str, default=None, help='Path to resume fine-tuning from')
    parser.add_argument('--lambda_base', type=float, default=0.7, help='Weight for base model loss during fine-tuning')
    parser.add_argument('--lambda_refinement', type=float, default=1.0, help='Weight for refinement loss during fine-tuning')
    
    # Refinement loss weights
    parser.add_argument('--lambda_recon', type=float, default=1.0, help='Weight for reconstruction loss in refinement')
    parser.add_argument('--lambda_magnitude', type=float, default=0.1, help='Weight for refinement magnitude loss')
    parser.add_argument('--lambda_improve', type=float, default=0.5, help='Weight for progressive improvement loss')
    
    # Base model loss weights
    parser.add_argument('--lambda_l1', type=float, default=1.0, help='Weight for L1 loss')
    parser.add_argument('--lambda_ssim', type=float, default=0.1, help='Weight for SSIM loss')
    parser.add_argument('--lambda_perceptual', type=float, default=0.1, help='Weight for perceptual loss')
    parser.add_argument('--lambda_diversity', type=float, default=0.05, help='Weight for diversity loss')
    parser.add_argument('--lambda_degradation', type=float, default=0.1, help='Weight for degradation consistency loss')
    
    # Logging and saving
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory for tensorboard logs')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory for saving models')
    parser.add_argument('--output_dir', type=str, default='results', help='Directory to save evaluation results')
    parser.add_argument('--log_interval', type=int, default=10, help='Logging interval (in iterations)')
    parser.add_argument('--save_interval', type=int, default=10, help='Model saving interval (in epochs)')
    parser.add_argument('--val_interval', type=int, default=1, help='Validation interval (in epochs)')
    parser.add_argument('--vis_samples', type=int, default=8, help='Number of samples to visualize')
    
    # Save epoch images option
    parser.add_argument('--save_epoch_images', action='store_true', 
                        help='Save sample images after each epoch during training/validation')
    
    # Evaluation parameters
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to model checkpoint for evaluation')
    parser.add_argument('--progressive_checkpoint', type=str, default=None, 
                       help='Path to progressive model checkpoint for evaluation')
    parser.add_argument('--compare_with_base', action='store_true', 
                       help='Compare progressive model with base model during evaluation')
    parser.add_argument('--save_images', action='store_true', help='Save restored images')
    parser.add_argument('--save_hypotheses', action='store_true', help='Save individual hypotheses')
    parser.add_argument('--save_degradation_maps', action='store_true', help='Save degradation maps')
    parser.add_argument('--save_refinement', action='store_true', help='Save refinement maps and residual degradation maps Mr maps')
    parser.add_argument('--visualize', action='store_true', help='Create visualization figures')
    
    # Ablation study parameters
    parser.add_argument('--ablation_type', type=str, default=None, choices=[
        'no_degradation_maps', 'single_hypothesis', 'fusion_type', 'no_diversity_loss',
        'decoder_types', 'no_refinement', 'refinement_magnitude'
    ], help='Type of ablation study to run')
    
    # Ablation training parameters
    parser.add_argument('--run_all_ablations', action='store_true', 
                        help='Run all ablation types sequentially')
    parser.add_argument('--resume_from_ablation', type=str, default=None,
                        help='Resume ablation studies from this ablation type')
    parser.add_argument('--use_improved_ablation', action='store_true',
                        help='Use the improved ablation manager for better stability')
    
    args = parser.parse_args()
    return args


def setup_model(args):
    """Create and initialize the model based on mode"""
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    if args.mode in ['train', 'eval', 'ablation'] or args.compare_with_base:
        # Create base model
        base_model = DGMHRN(
            in_channels=3,
            base_channels=args.base_channels,
            num_downs=args.num_downs,
            num_degradation_types=args.num_degradation_types,
            norm_type=args.norm_type,
            activation=args.activation,
            fusion_type=args.fusion_type
        ).to(device)
        
        # Print model info
        print(f"Base model created with {count_parameters(base_model):,} trainable parameters")
        
        # Load checkpoint if specified
        if args.mode == 'train' and args.resume:
            checkpoint = load_checkpoint(args.resume, base_model)
            print(f"Resumed base model from checkpoint: {args.resume}")
            if 'epoch' in checkpoint:
                args.start_epoch = checkpoint['epoch'] + 1
                print(f"Starting from epoch {args.start_epoch}")
            if 'best_psnr' in checkpoint:
                best_psnr = checkpoint['best_psnr']
                print(f"Previous best PSNR: {best_psnr:.2f}")
        elif args.mode in ['eval', 'ablation'] and args.checkpoint:
            checkpoint = load_checkpoint(args.checkpoint, base_model)
            print(f"Loaded base model checkpoint for evaluation: {args.checkpoint}")
        
        # For evaluation with both models
        if args.compare_with_base:
            model = base_model
        elif args.mode in ['train', 'eval', 'ablation']:
            model = base_model
    
    # Create or load progressive model if needed
    if args.mode in ['train_progressive', 'train_refinement', 'finetune', 'eval'] or args.progressive_checkpoint:
        if args.mode == 'train_progressive':
            # Base model will be loaded inside the train_progressive_restoration function
            model = None
        elif args.mode == 'train_refinement':
            # Load base model and create progressive model
            if args.base_model_path:
                base_model = DGMHRN(
                    in_channels=3,
                    base_channels=args.base_channels,
                    num_downs=args.num_downs,
                    num_degradation_types=args.num_degradation_types,
                    norm_type=args.norm_type,
                    activation=args.activation,
                    fusion_type=args.fusion_type
                ).to(device)
                
                checkpoint = load_checkpoint(args.base_model_path, base_model)
                print(f"Loaded base model from {args.base_model_path} for refinement training")
                
                # Create progressive model with the loaded base model
                model = TIDE(
                    base_model=base_model,
                    num_degradation_types=args.num_degradation_types,
                    norm_type=args.norm_type,
                    activation=args.activation
                ).to(device)
                
                # Freeze base model parameters
                for param in model.base_model.parameters():
                    param.requires_grad = False
                
                # Load refinement checkpoint if resuming
                if args.refinement_resume:
                    checkpoint = load_checkpoint(args.refinement_resume, model)
                    print(f"Resumed refinement training from: {args.refinement_resume}")
            else:
                raise ValueError("Base model path must be provided for refinement training")
                
        elif args.mode == 'finetune':
            # Load progressive model checkpoint with refinement components
            if args.finetune_resume:
                # Create progressive model
                base_model = DGMHRN(
                    in_channels=3,
                    base_channels=args.base_channels,
                    num_downs=args.num_downs,
                    num_degradation_types=args.num_degradation_types,
                    norm_type=args.norm_type,
                    activation=args.activation,
                    fusion_type=args.fusion_type
                ).to(device)
                
                model = TIDE(
                    base_model=base_model,
                    num_degradation_types=args.num_degradation_types,
                    norm_type=args.norm_type,
                    activation=args.activation
                ).to(device)
                
                # Load fine-tuning checkpoint
                checkpoint = load_checkpoint(args.finetune_resume, model)
                print(f"Resumed fine-tuning from: {args.finetune_resume}")
            else:
                raise ValueError("Fine-tuning requires a checkpoint to resume from")
                
        elif args.mode == 'eval' or args.progressive_checkpoint:
            # Load progressive model for evaluation
            if args.progressive_checkpoint:
                # First create base model
                base_model = DGMHRN(
                    in_channels=3,
                    base_channels=args.base_channels,
                    num_downs=args.num_downs,
                    num_degradation_types=args.num_degradation_types,
                    norm_type=args.norm_type,
                    activation=args.activation,
                    fusion_type=args.fusion_type
                ).to(device)
                
                # Create progressive model with the base model
                model = TIDE(
                    base_model=base_model,
                    num_degradation_types=args.num_degradation_types,
                    norm_type=args.norm_type,
                    activation=args.activation
                ).to(device)
                
                # Load progressive checkpoint
                checkpoint = load_checkpoint(args.progressive_checkpoint, model)
                print(f"Loaded progressive model from: {args.progressive_checkpoint}")
            else:
                # If no progressive checkpoint is specified, use the base model
                if not args.compare_with_base:  # Not handled above
                    model = base_model
    
    if args.mode not in ['train_progressive']:
        # Print parameter count
        if isinstance(model, TIDE):
            base_params = count_parameters(model.base_model)
            refine_params = count_parameters(model) - base_params
            print(f"Progressive model: {count_parameters(model):,} total parameters")
            print(f"  - Base model: {base_params:,} parameters")
            print(f"  - Refinement components: {refine_params:,} parameters")
        else:
            print(f"Model: {count_parameters(model):,} parameters")
    
    return model, device


def run_train(args):
    """Run base model training with the provided arguments"""
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Create directories
    create_directory(args.log_dir)
    create_directory(args.save_dir)
    if args.save_epoch_images:
        create_directory(os.path.join(args.save_dir, 'images'))
    
    # Create model
    model, device = setup_model(args)
    
    # Create dataloaders
    train_loader, val_loader = get_dataloaders(
        root_dir=args.data_dir,
        batch_size=args.batch_size,
        img_size=args.img_size,
        crop_size=args.crop_size,
        num_workers=args.num_workers
    )
    
    # Run training
    print(f"Starting base model training for {args.num_epochs} epochs")
    train_model(model, train_loader, val_loader, device, args)
    print("Base model training completed!")


def run_train_progressive(args):
    """Run the complete progressive training pipeline"""
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Create directories
    create_directory(args.log_dir)
    create_directory(args.save_dir)
    if args.save_epoch_images:
        create_directory(os.path.join(args.save_dir, 'images'))
    
    # Create dataloaders
    train_loader, val_loader = get_dataloaders(
        root_dir=args.data_dir,
        batch_size=args.batch_size,
        img_size=args.img_size,
        crop_size=args.crop_size,
        num_workers=args.num_workers
    )
    
    # Check if base model path is provided
    if not args.base_model_path:
        raise ValueError("Base model path must be provided for progressive training")
    
    # Get device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Run progressive training pipeline
    print(f"Starting two-stage restoration training pipeline")
    progressive_model = train_progressive_restoration(
        args.base_model_path, train_loader, val_loader, device, args
    )
    print("Progressive training completed!")


def run_train_refinement(args):
    """Run only the refinement training stage"""
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Create directories
    create_directory(args.log_dir)
    create_directory(args.save_dir)
    if args.save_epoch_images:
        create_directory(os.path.join(args.save_dir, 'images'))
    
    # Check if base model path is provided
    if not args.base_model_path:
        raise ValueError("Base model path must be provided for refinement training")
    
    # Create model
    model, device = setup_model(args)
    
    # Create dataloaders
    train_loader, val_loader = get_dataloaders(
        root_dir=args.data_dir,
        batch_size=args.batch_size,
        img_size=args.img_size,
        crop_size=args.crop_size,
        num_workers=args.num_workers
    )
    
    # Run refinement training
    print(f"Starting refinement training for {args.refinement_epochs} epochs")
    refined_model = train_refinement(model.base_model, train_loader, val_loader, device, args)
    print("Refinement training completed!")


def run_finetune(args):
    """Run end-to-end fine-tuning of the progressive model"""
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Create directories
    create_directory(args.log_dir)
    create_directory(args.save_dir)
    if args.save_epoch_images:
        create_directory(os.path.join(args.save_dir, 'images'))
    
    # Check if fine-tuning checkpoint is provided
    if not args.finetune_resume:
        raise ValueError("Fine-tuning requires a checkpoint to resume from")
    
    # Create model
    model, device = setup_model(args)
    
    # Create dataloaders
    train_loader, val_loader = get_dataloaders(
        root_dir=args.data_dir,
        batch_size=args.batch_size,
        img_size=args.img_size,
        crop_size=args.crop_size,
        num_workers=args.num_workers
    )
    
    # Run fine-tuning
    print(f"Starting end-to-end fine-tuning for {args.finetune_epochs} epochs")
    finetuned_model = finetune_progressive_model(model, train_loader, val_loader, device, args)
    print("Fine-tuning completed!")


def run_eval(args):
    """Run evaluation with the provided arguments"""
    # Create output directory
    create_directory(args.output_dir)
    
    # Create model and load checkpoint
    model, device = setup_model(args)
    
    # Create dataloader (only validation)
    _, val_loader = get_dataloaders(
        root_dir=args.data_dir,
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_workers=args.num_workers,
        augment=False
    )
    
    # Check if comparing with base model
    if args.compare_with_base and args.progressive_checkpoint:
        # Evaluate base model
        print("Evaluating base model...")
        base_metrics = evaluate(model, val_loader, device, args, is_progressive=False)
        
        # Print base model results
        print("\nBase Model Evaluation Results:")
        for k, v in base_metrics.items():
            print(f"{k}: {v:.4f}")
        
        # Create output directory for progressive model
        progressive_output_dir = os.path.join(args.output_dir, 'progressive')
        create_directory(progressive_output_dir)
        
        # Set the progressive checkpoint for evaluation
        args.checkpoint = args.progressive_checkpoint
        
        # Create and load progressive model
        base_model = DGMHRN(
            in_channels=3,
            base_channels=args.base_channels,
            num_downs=args.num_downs,
            num_degradation_types=args.num_degradation_types,
            norm_type=args.norm_type,
            activation=args.activation,
            fusion_type=args.fusion_type
        ).to(device)
        
        progressive_model = TIDE(
            base_model=base_model,
            num_degradation_types=args.num_degradation_types,
            norm_type=args.norm_type,
            activation=args.activation
        ).to(device)
        
        checkpoint = load_checkpoint(args.progressive_checkpoint, progressive_model)
        print(f"Loaded progressive model from: {args.progressive_checkpoint}")
        
        # Temporarily update args for progressive evaluation
        orig_output_dir = args.output_dir
        args.output_dir = progressive_output_dir
        
        # Evaluate progressive model
        print("\nEvaluating progressive model...")
        progressive_metrics = evaluate(progressive_model, val_loader, device, args, is_progressive=True)
        
        # Restore original output directory
        args.output_dir = orig_output_dir
        
        # Print progressive model results
        print("\nProgressive Model Evaluation Results:")
        for k, v in progressive_metrics.items():
            print(f"{k}: {v:.4f}")
        
        # Calculate and print improvements
        print("\nImprovements:")
        print(f"PSNR: +{progressive_metrics['psnr'] - base_metrics['psnr']:.4f} dB")
        print(f"SSIM: +{progressive_metrics['ssim'] - base_metrics['ssim']:.4f}")
        
        # Save comparison results
        with open(os.path.join(args.output_dir, 'comparison_results.txt'), 'w') as f:
            f.write("Base Model vs Progressive Model Comparison\n")
            f.write("=======================================\n\n")
            
            f.write("Base Model Results:\n")
            for k, v in base_metrics.items():
                f.write(f"{k}: {v:.4f}\n")
            
            f.write("\nProgressive Model Results:\n")
            for k, v in progressive_metrics.items():
                f.write(f"{k}: {v:.4f}\n")
            
            f.write("\nImprovements:\n")
            f.write(f"PSNR: +{progressive_metrics['psnr'] - base_metrics['psnr']:.4f} dB\n")
            f.write(f"SSIM: +{progressive_metrics['ssim'] - base_metrics['ssim']:.4f}\n")
        
        # Return both metrics
        return base_metrics, progressive_metrics
    
    else:
        # Run evaluation for a single model
        print("Starting evaluation...")
        is_progressive = args.progressive_checkpoint is not None
        metrics = evaluate(model, val_loader, device, args, is_progressive=is_progressive)
        
        # Print results
        print("\nEvaluation Results:")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")
        
        return metrics


def run_ablation(args):
    """Run ablation studies"""
    # Check if ablation type is specified
    if args.ablation_type is None and not args.run_all_ablations:
        raise ValueError("Must specify either --ablation_type or --run_all_ablations")
    
    # Create output directory
    create_directory(args.output_dir)
    
    # Use improved ablation manager if specified
    if args.use_improved_ablation:
        if args.run_all_ablations:
            # List of all ablation types
            all_ablations = [
                'no_degradation_maps', 
                'single_hypothesis', 
                'fusion_type', 
                'no_diversity_loss',
                'decoder_types',
                'no_refinement',
                'refinement_magnitude'
            ]
            
            # Check if we should resume from a specific ablation
            if args.resume_from_ablation:
                if args.resume_from_ablation not in all_ablations:
                    raise ValueError(f"Invalid ablation type to resume from: {args.resume_from_ablation}")
                
                start_idx = all_ablations.index(args.resume_from_ablation)
                all_ablations = all_ablations[start_idx:]
            
            # Run each ablation type
            for ablation_type in all_ablations:
                print(f"\n{'='*50}")
                print(f"Running ablation study: {ablation_type}")
                print(f"{'='*50}\n")
                
                # Update args with current ablation type
                args.ablation_type = ablation_type
                
                try:
                    # Run this ablation with the improved manager
                    run_improved_ablation_study(args)
                except Exception as e:
                    print(f"Error running ablation {ablation_type}: {e}")
                    import traceback
                    traceback.print_exc()
                    print("Continuing with next ablation...")
                    continue
        else:
            # Run single ablation study with improved manager
            run_improved_ablation_study(args)
    else:
        # Use original ablation implementation
        if args.run_all_ablations:
            # List of all ablation types
            all_ablations = [
                'no_degradation_maps', 
                'single_hypothesis', 
                'fusion_type', 
                'no_diversity_loss',
                'decoder_types',
                'no_refinement',
                'refinement_magnitude'
            ]
            
            # Check if we should resume from a specific ablation
            if args.resume_from_ablation:
                if args.resume_from_ablation not in all_ablations:
                    raise ValueError(f"Invalid ablation type to resume from: {args.resume_from_ablation}")
                
                start_idx = all_ablations.index(args.resume_from_ablation)
                all_ablations = all_ablations[start_idx:]
            
            # Run each ablation type
            for ablation_type in all_ablations:
                print(f"\n{'='*50}")
                print(f"Running ablation study: {ablation_type}")
                print(f"{'='*50}\n")
                
                # Update args with current ablation type
                args.ablation_type = ablation_type
                
                try:
                    # Run this ablation
                    run_ablation_study(args)
                except Exception as e:
                    print(f"Error running ablation {ablation_type}: {e}")
                    import traceback
                    traceback.print_exc()
                    print("Continuing with next ablation...")
                    continue
        else:
            # Run single ablation study with original manager
            run_ablation_study(args)


def main():
    # Parse command line arguments
    args = parse_args()
    
    # Run in the specified mode
    if args.mode == 'train':
        run_train(args)
    elif args.mode == 'train_progressive':
        run_train_progressive(args)
    elif args.mode == 'train_refinement':
        run_train_refinement(args)
    elif args.mode == 'finetune':
        run_finetune(args)
    elif args.mode == 'eval':
        run_eval(args)
    elif args.mode == 'ablation':
        run_ablation(args)
    else:
        print(f"Unknown mode: {args.mode}")


if __name__ == '__main__':
    main()
