"""
Configuration file for ablation studies, defining parameters for efficient training
"""

# Base configuration for all ablation studies
BASE_CONFIG = {
    # Model parameters
    'base_channels': 64,
    'num_downs': 5,
    'num_degradation_types': 4,
    'norm_type': 'instance',
    'activation': 'leaky_relu',
    'fusion_type': 'learned',
    
    # Dataset parameters
    'img_size': 128,
    'crop_size': 128,
    'batch_size': 8,  # Reduced batch size for stability
    
    # Training parameters
    'num_epochs': 32,
    'lr': 1e-5,  # Reduced learning rate for stability
    'weight_decay': 1e-4,
    'lr_cycle_epochs': 16,
    'lr_cycle_mult': 1,
    'grad_clip': 0.1,  # More aggressive clipping for stability
    'mixed_precision': True,
    
    # Validation parameters
    'val_interval': 1,
    'save_interval': 8,
    
    # Refinement parameters
    'refinement_epochs': 32,
    'refinement_lr': 5e-6,  # Reduced learning rate for stability
    
    # Loss weights
    'lambda_l1': 1.0,
    'lambda_ssim': 0.1,
    'lambda_perceptual': 0.1,
    'lambda_diversity': 0.05,
    'lambda_degradation': 0.1,
    
    # Refinement loss weights
    'lambda_recon': 1.0,
    'lambda_magnitude': 0.1,
    'lambda_improve': 0.5,
    
    # Logging parameters
    'log_interval': 10,
    'vis_samples': 4,
    'save_epoch_images': False,
    
    # Use improved ablation manager
    'use_improved_ablation': True
}

# Specific configurations for each ablation type
ABLATION_CONFIGS = {
    'no_degradation_maps': {
        # Use extremely conservative training parameters for this unstable model
        'lr': 5e-6,
        'batch_size': 4,
        'grad_clip': 0.05,
        'weight_decay': 5e-4,
        'num_epochs': 50  # Train longer since this model learns slower
    },
    
    'single_hypothesis': {
        # Using only one decoder, so we can reduce channels
        'base_channels': 48,
        'num_degradation_types': 1,
        'lr': 5e-6,  # Lower learning rate for stability
        'grad_clip': 0.1
    },
    
    'fusion_type': {
        # Multiple runs will be done, one for each fusion type
        'lr': 5e-6,  # Lower learning rate for stability
        'grad_clip': 0.1
    },
    
    'no_diversity_loss': {
        # One run with diversity loss, one without
        # Use standard parameters for this ablation
    },
    
    'decoder_types': {
        # Multiple runs will be done, one for each decoder combination
        'lr': 5e-6,  # Lower learning rate for stability
        'grad_clip': 0.1
    },
    
    'no_refinement': {
        # One run with refinement, one without
        'lr': 5e-6,  # Lower learning rate for stability
        'grad_clip': 0.1,
        'refinement_lr': 5e-6
    },
    
    'refinement_magnitude': {
        # Multiple runs with different refinement scales
        'refinement_epochs': 25,  # Can be shorter since only tuning scaling factors
        'lr': 5e-6,  # Lower learning rate for stability
        'refinement_lr': 5e-6,
        'grad_clip': 0.1
    }
}

def get_config_for_ablation(ablation_type):
    """
    Get configuration for a specific ablation type
    
    Args:
        ablation_type: Type of ablation study
        
    Returns:
        Dictionary of configuration parameters
    """
    # Start with base config
    config = BASE_CONFIG.copy()
    
    # Update with ablation-specific config if it exists
    if ablation_type in ABLATION_CONFIGS:
        config.update(ABLATION_CONFIGS[ablation_type])
    
    return config


def apply_config_to_args(args, config):
    """
    Apply configuration to command line arguments
    
    Args:
        args: ArgumentParser arguments
        config: Configuration dictionary
        
    Returns:
        Updated args with config values
    """
    for key, value in config.items():
        if hasattr(args, key):
            setattr(args, key, value)
    
    return args
