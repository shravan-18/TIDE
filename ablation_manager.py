import os
import sys
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
import cv2
from PIL import Image
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter

from model import DGMHRN, TIDE
from ablation_models import create_ablation_model
from dataset import get_dataloaders
from utils import create_directory, load_checkpoint, save_checkpoint, count_parameters
from trainer import validate
from losses import CombinedLoss, RefinementLoss


class ImprovedAblationManager:
    """Enhanced ablation manager with improved stability and error handling"""
    
    def __init__(self, args):
        """Initialize the ablation manager with improved error handling"""
        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        
        # Set random seed for reproducibility
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        # Create ablation output directory - FIX: Remove the 'ablation_' prefix
        self.ablation_dir = os.path.join(args.output_dir, f'{args.ablation_type}')
        create_directory(self.ablation_dir)
        
        # Create visualization directory
        self.vis_dir = os.path.join(self.ablation_dir, 'visualizations')
        create_directory(self.vis_dir)
        
        # Create tensorboard directory
        self.tb_dir = os.path.join(self.ablation_dir, 'tensorboard')
        create_directory(self.tb_dir)
        
        # Results dictionary to store metrics from all runs
        self.results = {}
        
        # Configure ablation-specific parameters
        self.configure_ablation()
        
    def configure_ablation(self):
        """Configure parameters specific to the requested ablation type"""
        
        # Base model parameters for all ablations
        self.model_params = {
            'in_channels': 3,
            'base_channels': self.args.base_channels,
            'num_downs': self.args.num_downs,
            'num_degradation_types': self.args.num_degradation_types,
            'norm_type': self.args.norm_type,
            'activation': self.args.activation,
            'fusion_type': self.args.fusion_type
        }
        
        # Configure based on ablation type
        if self.args.ablation_type == 'no_degradation_maps':
            self.variants = ['no_degradation_maps']
            self.variant_names = ['No Degradation Maps']
            
        elif self.args.ablation_type == 'single_hypothesis':
            self.variants = ['single_hypothesis']
            self.variant_names = ['Single Hypothesis (Color)']
            
        elif self.args.ablation_type == 'fusion_type':
            self.variants = ['direct', 'learned', 'attention']
            self.variant_names = ['Direct Fusion', 'Learned Fusion', 'Attention Fusion']
            
        elif self.args.ablation_type == 'no_diversity_loss':
            self.variants = ['with_diversity', 'no_diversity']
            self.variant_names = ['With Diversity Loss', 'No Diversity Loss']
            # Modify loss weights for no_diversity variant
            self.loss_weights = {
                'with_diversity': {'lambda_diversity': self.args.lambda_diversity},
                'no_diversity': {'lambda_diversity': 0.0}
            }
            
        elif self.args.ablation_type == 'decoder_types':
            # Define decoder combinations to test
            self.variants = [
                'color_only',
                'color_contrast',
                'color_detail',
                'color_denoise',
                'color_contrast_detail',
                'color_contrast_denoise',
                'color_detail_denoise',
                'all_decoders'
            ]
            
            self.variant_names = [
                'Color Only',
                'Color + Contrast',
                'Color + Detail',
                'Color + Denoise',
                'Color + Contrast + Detail',
                'Color + Contrast + Denoise',
                'Color + Detail + Denoise',
                'All Decoders'
            ]
            
            # Map variant names to decoder combinations
            self.decoder_combinations = {
                'color_only': ['color'],
                'color_contrast': ['color', 'contrast'],
                'color_detail': ['color', 'detail'],
                'color_denoise': ['color', 'denoise'],
                'color_contrast_detail': ['color', 'contrast', 'detail'],
                'color_contrast_denoise': ['color', 'contrast', 'denoise'],
                'color_detail_denoise': ['color', 'detail', 'denoise'],
                'all_decoders': ['color', 'contrast', 'detail', 'denoise']
            }
            
        elif self.args.ablation_type == 'no_refinement':
            self.variants = ['with_refinement', 'no_refinement']
            self.variant_names = ['With Progressive Refinement Stage', 'No Refinement (Base Only)']
            
        elif self.args.ablation_type == 'refinement_magnitude':
            self.variants = ['scale_0.2', 'scale_0.5', 'scale_1.0', 'scale_2.0', 'scale_5.0']
            self.variant_names = ['Scale 0.2x', 'Scale 0.5x', 'Scale 1.0x (Default)', 'Scale 2.0x', 'Scale 5.0x']
            self.refinement_scales = {
                'scale_0.2': 0.2,
                'scale_0.5': 0.5,
                'scale_1.0': 1.0,
                'scale_2.0': 2.0,
                'scale_5.0': 5.0
            }
    
    def create_model_for_variant(self, variant):
        """
        Create the appropriate model for a specific ablation variant
        
        Args:
            variant: The specific ablation variant
            
        Returns:
            Initialized model
        """
        model_params = self.model_params.copy()
        
        if self.args.ablation_type == 'fusion_type':
            # Set fusion type
            model_params['fusion_type'] = variant
            model = create_ablation_model('fusion_type', model_params)
            
        elif self.args.ablation_type == 'decoder_types':
            # Set decoder types
            model_params['decoder_types'] = self.decoder_combinations[variant]
            model = create_ablation_model('decoder_types', model_params)
            
        elif self.args.ablation_type == 'no_degradation_maps':
            model = create_ablation_model('no_degradation_maps', model_params)
            
        elif self.args.ablation_type == 'single_hypothesis':
            model = create_ablation_model('single_hypothesis', model_params)
            
        elif self.args.ablation_type == 'no_diversity_loss':
            # Standard model, diversity loss will be configured in training
            model = create_ablation_model(None, model_params)
            
        elif self.args.ablation_type == 'no_refinement':
            # Add variant parameter
            model_params['variant'] = variant
            model = create_ablation_model('no_refinement', model_params)
            
        elif self.args.ablation_type == 'refinement_magnitude':
            # Add refinement scale parameter
            model_params['refinement_scale'] = self.refinement_scales[variant]
            model = create_ablation_model('refinement_magnitude', model_params)
            
        else:
            raise ValueError(f"Unknown ablation type: {self.args.ablation_type}")
        
        return model.to(self.device)
    
    def get_loss_for_variant(self, variant):
        """
        Get the appropriate loss function for a specific variant
        
        Args:
            variant: The specific ablation variant
            
        Returns:
            Loss function
        """
        if self.args.ablation_type == 'no_diversity_loss':
            # Adjust diversity loss weight
            lambda_diversity = self.loss_weights[variant]['lambda_diversity']
            
            return CombinedLoss(
                lambda_l1=self.args.lambda_l1,
                lambda_ssim=self.args.lambda_ssim,
                lambda_perceptual=self.args.lambda_perceptual,
                lambda_diversity=lambda_diversity,
                lambda_degradation=self.args.lambda_degradation,
                use_simplified_perceptual=True  # Use simplified perceptual loss for stability
            ).to(self.device)
        else:
            # Standard loss function for other ablations
            return CombinedLoss(
                lambda_l1=self.args.lambda_l1,
                lambda_ssim=self.args.lambda_ssim,
                lambda_perceptual=self.args.lambda_perceptual,
                lambda_diversity=self.args.lambda_diversity,
                lambda_degradation=self.args.lambda_degradation,
                use_simplified_perceptual=True  # Use simplified perceptual loss for stability
            ).to(self.device)
    
    def save_sample_images(self, model, val_loader, variant, epoch=-1):
        """
        Save sample images for visualization
        
        Args:
            model: The model to generate images
            val_loader: Validation data loader
            variant: The specific ablation variant
            epoch: Current epoch (if during training) or -1 for final evaluation
        """
        # Create directories
        save_dir = os.path.join(self.vis_dir, variant, 'samples')
        create_directory(save_dir)
        
        # Get a batch of data
        batch = next(iter(val_loader))
        degraded = batch['degraded'].to(self.device)
        reference = batch['reference'].to(self.device)
        filenames = batch['filename']
        
        # Generate outputs
        model.eval()
        with torch.no_grad():
            outputs = model(degraded)
            
            # Handle potential NaN values
            if torch.isnan(outputs['restored_image']).any():
                outputs['restored_image'] = torch.nan_to_num(outputs['restored_image'], nan=0.0)
                outputs['restored_image'] = torch.clamp(outputs['restored_image'], 0.0, 1.0)
        
        # Save a few sample images
        for i in range(min(5, len(degraded))):
            # Create a grid of images
            images = []
            
            # Add degraded image
            images.append(degraded[i])
            
            # Add hypotheses if available
            if 'hypotheses' in outputs:
                for hyp in outputs['hypotheses']:
                    # Handle NaN values
                    if torch.isnan(hyp[i]).any():
                        hyp[i] = torch.nan_to_num(hyp[i], nan=0.0)
                        hyp[i] = torch.clamp(hyp[i], 0.0, 1.0)
                    images.append(hyp[i])
            
            # Add restored image
            images.append(outputs['restored_image'][i])
            
            # Add reference image
            images.append(reference[i])
            
            # If progressive model, add initial and refinement
            if 'initial_restoration' in outputs and 'refinement' in outputs:
                # Handle NaN values
                if torch.isnan(outputs['initial_restoration'][i]).any():
                    outputs['initial_restoration'][i] = torch.nan_to_num(outputs['initial_restoration'][i], nan=0.0)
                    outputs['initial_restoration'][i] = torch.clamp(outputs['initial_restoration'][i], 0.0, 1.0)
                
                if torch.isnan(outputs['refinement'][i]).any():
                    outputs['refinement'][i] = torch.nan_to_num(outputs['refinement'][i], nan=0.0)
                
                images.append(outputs['initial_restoration'][i])
                # Scale refinement for visibility
                refinement = outputs['refinement'][i] * 5.0 + 0.5
                refinement = torch.clamp(refinement, 0, 1)
                images.append(refinement)
            
            # Create grid
            grid = torch.stack(images)
            grid = vutils.make_grid(grid, nrow=1, padding=2, normalize=False)
            
            # Save grid
            epoch_str = f"epoch_{epoch}_" if epoch >= 0 else ""
            vutils.save_image(
                grid, 
                os.path.join(save_dir, f"{epoch_str}{filenames[i]}")
            )
            
            # Also save individual images for easier comparison
            if epoch < 0:  # Only for final evaluation
                individual_dir = os.path.join(save_dir, 'individual')
                create_directory(individual_dir)
                
                # Save degraded, restored, and reference
                vutils.save_image(
                    degraded[i],
                    os.path.join(individual_dir, f"degraded_{filenames[i]}")
                )
                vutils.save_image(
                    outputs['restored_image'][i],
                    os.path.join(individual_dir, f"restored_{filenames[i]}")
                )
                vutils.save_image(
                    reference[i],
                    os.path.join(individual_dir, f"reference_{filenames[i]}")
                )
    
    def save_degradation_visualizations(self, model, val_loader, variant):
        """
        Save visualizations of degradation maps
        
        Args:
            model: The model to generate images
            val_loader: Validation data loader
            variant: The specific ablation variant
        """
        # Create directory
        save_dir = os.path.join(self.vis_dir, variant, 'degradation_maps')
        create_directory(save_dir)
        
        # Get a batch of data
        batch = next(iter(val_loader))
        degraded = batch['degraded'].to(self.device)
        filenames = batch['filename']
        
        # Generate outputs
        model.eval()
        with torch.no_grad():
            outputs = model(degraded)
            
            # Handle potential NaN values
            if 'degradation_maps' in outputs and torch.isnan(outputs['degradation_maps']).any():
                outputs['degradation_maps'] = torch.nan_to_num(outputs['degradation_maps'], nan=0.0)
                outputs['degradation_maps'] = torch.clamp(outputs['degradation_maps'], 0.0, 1.0)
        
        # Save degradation maps for a few samples
        if 'degradation_maps' in outputs:
            for i in range(min(5, len(degraded))):
                # Get degradation maps
                deg_maps = outputs['degradation_maps'][i]
                
                # Create heatmap visualizations
                for j in range(deg_maps.size(0)):
                    # Convert to numpy
                    map_np = deg_maps[j].cpu().numpy()
                    
                    # Normalize to [0, 255]
                    map_min = map_np.min()
                    map_max = map_np.max()
                    if map_max > map_min:  # Avoid division by zero
                        map_np = (map_np - map_min) / (map_max - map_min) * 255
                    map_np = map_np.astype(np.uint8)
                    
                    # Apply colormap
                    try:
                        heatmap = cv2.applyColorMap(map_np, cv2.COLORMAP_JET)
                        
                        # Save as image
                        cv2.imwrite(
                            os.path.join(save_dir, f"{filenames[i]}_map_{j}.png"),
                            heatmap
                        )
                    except Exception as e:
                        print(f"Error creating heatmap: {e}")
                
                # Create combined visualization
                try:
                    combined = torch.cat([
                        degraded[i],
                        outputs['restored_image'][i]
                    ], dim=2)
                    
                    vutils.save_image(
                        combined,
                        os.path.join(save_dir, f"{filenames[i]}_combined.png")
                    )
                except Exception as e:
                    print(f"Error creating combined visualization: {e}")
    
    def save_refinement_visualizations(self, model, val_loader, variant):
        """
        Save visualizations of progressive refinement
        
        Args:
            model: The model to generate images
            val_loader: Validation data loader
            variant: The specific ablation variant
        """
        # Check if model has refinement capabilities
        if not hasattr(model, 'enable_refinement'):
            return
            
        # Create directory
        save_dir = os.path.join(self.vis_dir, variant, 'refinement')
        create_directory(save_dir)
        
        # Get a batch of data
        batch = next(iter(val_loader))
        degraded = batch['degraded'].to(self.device)
        reference = batch['reference'].to(self.device)
        filenames = batch['filename']
        
        try:
            # Generate outputs with refinement enabled
            model.eval()
            model.enable_refinement = True
            with torch.no_grad():
                outputs_with = model(degraded)
                
                # Handle potential NaN values
                if torch.isnan(outputs_with['restored_image']).any():
                    outputs_with['restored_image'] = torch.nan_to_num(outputs_with['restored_image'], nan=0.0)
                    outputs_with['restored_image'] = torch.clamp(outputs_with['restored_image'], 0.0, 1.0)
                
                if 'initial_restoration' in outputs_with and torch.isnan(outputs_with['initial_restoration']).any():
                    outputs_with['initial_restoration'] = torch.nan_to_num(outputs_with['initial_restoration'], nan=0.0)
                    outputs_with['initial_restoration'] = torch.clamp(outputs_with['initial_restoration'], 0.0, 1.0)
                
                if 'refinement' in outputs_with and torch.isnan(outputs_with['refinement']).any():
                    outputs_with['refinement'] = torch.nan_to_num(outputs_with['refinement'], nan=0.0)
            
            # Generate outputs with refinement disabled
            model.enable_refinement = False
            with torch.no_grad():
                outputs_without = model(degraded)
                
                # Handle potential NaN values
                if torch.isnan(outputs_without['restored_image']).any():
                    outputs_without['restored_image'] = torch.nan_to_num(outputs_without['restored_image'], nan=0.0)
                    outputs_without['restored_image'] = torch.clamp(outputs_without['restored_image'], 0.0, 1.0)
            
            # Re-enable refinement for future use
            model.enable_refinement = True
            
            # Save comparison visualizations
            for i in range(min(5, len(degraded))):
                try:
                    # Create a side-by-side comparison
                    images = [
                        degraded[i],                             # Degraded input
                        outputs_without['restored_image'][i],    # Without refinement
                        outputs_with['restored_image'][i],       # With refinement
                        reference[i]                             # Reference
                    ]
                    
                    # Create grid
                    grid = torch.stack(images)
                    grid = vutils.make_grid(grid, nrow=4, padding=2, normalize=False)
                    
                    # Save grid
                    vutils.save_image(
                        grid, 
                        os.path.join(save_dir, f"refinement_comparison_{filenames[i]}")
                    )
                    
                    # Save refinement map if available
                    if 'refinement' in outputs_with:
                        # Scale refinement for visibility
                        refinement = outputs_with['refinement'][i] * 5.0 + 0.5
                        refinement = torch.clamp(refinement, 0, 1)
                        
                        vutils.save_image(
                            refinement,
                            os.path.join(save_dir, f"refinement_map_{filenames[i]}")
                        )
                except Exception as e:
                    print(f"Error saving refinement visualization: {e}")
                    
        except Exception as e:
            print(f"Error in refinement visualization: {e}")
    
    def _train_epoch(self, model, train_loader, criterion, optimizer, scheduler, epoch):
        """Train for one epoch with enhanced stability"""
        model.train()
        
        # Metrics
        total_loss = 0
        total_psnr = 0
        total_ssim = 0
        valid_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for i, batch in enumerate(pbar):
            # Get data
            try:
                degraded = batch['degraded'].to(self.device)
                reference = batch['reference'].to(self.device)
                
                # Check for NaN inputs
                if torch.isnan(degraded).any() or torch.isnan(reference).any():
                    print("Warning: NaN detected in input batch, skipping")
                    continue
                
                # Forward pass with gradient scaling
                optimizer.zero_grad()
                
                # Use try/except to catch computation errors
                try:
                    outputs = model(degraded)
                    loss_dict = criterion(outputs, reference, degraded)
                    loss = loss_dict['total']
                    
                    # Check for NaN in loss
                    if torch.isnan(loss).any() or torch.isinf(loss).any():
                        print(f"Warning: NaN/Inf detected in loss at batch {i}, skipping")
                        continue
                    
                    # Backward pass with gradient clipping
                    loss.backward()
                    
                    # More aggressive gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.grad_clip)
                    
                    # Step optimizer and scheduler
                    optimizer.step()
                    scheduler.step()
                    
                    # Calculate metrics
                    with torch.no_grad():
                        restored = outputs['restored_image']
                        
                        # Ensure restored image doesn't have NaNs
                        if torch.isnan(restored).any():
                            restored = torch.nan_to_num(restored, nan=0.0)
                            restored = torch.clamp(restored, 0.0, 1.0)
                        
                        # Mean squared error
                        mse = torch.mean((restored - reference) ** 2).item()
                        if mse <= 0.0:
                            mse = 1e-8  # Prevent division by zero or negative numbers
                        
                        # PSNR
                        psnr = -10 * np.log10(mse)
                        
                        # SSIM from loss
                        ssim = 1.0 - loss_dict['ssim'].item()
                    
                    # Update metrics
                    total_loss += loss.item()
                    total_psnr += psnr
                    total_ssim += ssim
                    valid_batches += 1
                    
                    # Update progress bar
                    pbar.set_description(
                        f"Epoch {epoch} | Loss: {total_loss/max(1,valid_batches):.4f} | "
                        f"PSNR: {total_psnr/max(1,valid_batches):.2f} | SSIM: {total_ssim/max(1,valid_batches):.4f}"
                    )
                    
                except Exception as e:
                    print(f"Error during training batch {i}: {e}")
                    continue
                    
            except Exception as e:
                print(f"Error processing batch {i}: {e}")
                continue
        
        # Return average metrics
        if valid_batches > 0:
            return {
                'loss': total_loss / valid_batches,
                'psnr': total_psnr / valid_batches,
                'ssim': total_ssim / valid_batches,
                'valid_batches': valid_batches,
                'total_batches': len(train_loader)
            }
        else:
            print("Warning: No valid batches in epoch")
            return {
                'loss': float('nan'),
                'psnr': float('nan'),
                'ssim': float('nan'),
                'valid_batches': 0,
                'total_batches': len(train_loader)
            }
    
    def _validate(self, model, val_loader, criterion, epoch):
        """Validate the model with enhanced stability"""
        model.eval()
        
        # Use the more robust validation function from trainer.py
        try:
            val_metrics = validate(
                model=model,
                val_loader=val_loader,
                criterion=criterion,
                epoch=epoch,
                device=self.device,
                args=self.args
            )
            return val_metrics
        except Exception as e:
            print(f"Error during validation: {e}")
            # Return dummy metrics
            return {
                'loss': float('nan'),
                'psnr': float('nan'),
                'ssim': float('nan')
            }
    
    def _train_with_stability(self, model, train_loader, val_loader, criterion, variant_dir):
        """Train model with enhanced stability measures"""
        
        # Create optimizer with conservative parameters
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.args.lr,
            weight_decay=max(self.args.weight_decay, 1e-4),  # Increase regularization
            eps=1e-8  # Increased epsilon for numerical stability
        )
        
        # Use a learning rate scheduler with warm-up
        from torch.optim.lr_scheduler import OneCycleLR
        scheduler = OneCycleLR(
            optimizer,
            max_lr=self.args.lr,
            total_steps=self.args.num_epochs * len(train_loader),
            pct_start=0.3,  # 30% warm-up
            div_factor=10.0,
            final_div_factor=100.0,
            anneal_strategy='cos'
        )
        
        # Create a new writer for this variant - FIX: Use variant name to avoid path duplication
        variant_name = os.path.basename(variant_dir)
        writer_dir = os.path.join(self.tb_dir, variant_name)
        create_directory(writer_dir)
        writer = SummaryWriter(log_dir=writer_dir)
        
        # Track metrics
        best_psnr = 0
        best_metrics = None
        patience = 5  # Early stopping patience
        patience_counter = 0
        
        # Training loop with early stopping
        for epoch in range(self.args.num_epochs):
            # Train for one epoch
            train_metrics = self._train_epoch(
                model=model,
                train_loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch
            )
            
            # Log training metrics
            if not np.isnan(train_metrics['psnr']):
                writer.add_scalar('train/loss', train_metrics['loss'], epoch)
                writer.add_scalar('train/psnr', train_metrics['psnr'], epoch)
                writer.add_scalar('train/ssim', train_metrics['ssim'], epoch)
            
            # Validate
            val_metrics = self._validate(
                model=model,
                val_loader=val_loader,
                criterion=criterion,
                epoch=epoch
            )
            
            # Log validation metrics
            if not np.isnan(val_metrics['psnr']):
                writer.add_scalar('val/loss', val_metrics['loss'], epoch)
                writer.add_scalar('val/psnr', val_metrics['psnr'], epoch)
                writer.add_scalar('val/ssim', val_metrics['ssim'], epoch)
            
            # Check for NaN in metrics
            if np.isnan(val_metrics['psnr']):
                print(f"Warning: NaN detected in validation metrics at epoch {epoch}")
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Stopping early due to {patience} consecutive NaN epochs")
                    break
                continue
            else:
                patience_counter = 0  # Reset counter if we get valid metrics
            
            # Save visualization samples
            if epoch % 5 == 0:
                self.save_sample_images(model, val_loader, variant_name, epoch)
            
            # Save best model
            if val_metrics['psnr'] > best_psnr:
                best_psnr = val_metrics['psnr']
                best_metrics = val_metrics.copy()
                
                # Save checkpoint
                checkpoints_dir = os.path.join(variant_dir, 'checkpoints')
                create_directory(checkpoints_dir)
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    metrics=val_metrics,
                    best_psnr=best_psnr,
                    path=os.path.join(checkpoints_dir, 'best_model.pth')
                )
                print(f"Saved best model with PSNR {best_psnr:.2f}")
        
        # Close writer
        writer.close()
        
        # Return best metrics or final metrics if no valid ones found
        if best_metrics is not None:
            return best_metrics
        else:
            return val_metrics
    
    def run_variant(self, variant, train_loader, val_loader):
        """
        Run training for a specific ablation variant
        
        Args:
            variant: The specific ablation variant
            train_loader: Training data loader
            val_loader: Validation data loader
            
        Returns:
            Dictionary of results
        """
        print(f"\n{'='*50}")
        print(f"Running ablation variant: {variant}")
        print(f"{'='*50}\n")
        
        # Create variant-specific directories
        variant_dir = os.path.join(self.ablation_dir, variant)
        create_directory(variant_dir)
        
        # Set a lower learning rate for no_degradation_maps which has stability issues
        if self.args.ablation_type == 'no_degradation_maps':
            original_lr = self.args.lr
            self.args.lr = min(1e-5, self.args.lr)  # Lower learning rate
            print(f"Using reduced learning rate for no_degradation_maps: {self.args.lr}")
        
        try:
            # Create model for this variant
            model = self.create_model_for_variant(variant)
            
            # Count parameters
            num_params = count_parameters(model)
            print(f"Model has {num_params:,} trainable parameters")
            
            # Get criterion
            criterion = self.get_loss_for_variant(variant)
            
            # Train model with improved stability
            metrics = self._train_with_stability(model, train_loader, val_loader, criterion, variant_dir)
            
            # Generate visualizations
            self.save_sample_images(model, val_loader, variant)
            self.save_degradation_visualizations(model, val_loader, variant)
            
            # Generate refinement visualizations if applicable
            if hasattr(model, 'enable_refinement') or self.args.ablation_type == 'no_refinement':
                self.save_refinement_visualizations(model, val_loader, variant)
            
            print(f"\nVariant {variant} results:")
            for k, v in metrics.items():
                print(f"{k}: {v:.4f}")
            
            # Restore original learning rate if modified
            if self.args.ablation_type == 'no_degradation_maps':
                self.args.lr = original_lr
                
            return metrics
            
        except Exception as e:
            print(f"Error running variant {variant}: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Restore original learning rate if modified
            if self.args.ablation_type == 'no_degradation_maps':
                self.args.lr = original_lr
                
            # Return dummy metrics
            return {
                'psnr': 0.0,
                'ssim': 0.0,
                'loss': 999.0,
                'error': str(e)
            }
    
    def run_all_variants(self):
        """Run all variants for the specified ablation type"""
        # Create dataloaders
        train_loader, val_loader = get_dataloaders(
            root_dir=self.args.data_dir,
            batch_size=self.args.batch_size,
            img_size=self.args.img_size,
            crop_size=self.args.crop_size,
            num_workers=self.args.num_workers
        )
        
        # Run each variant
        for i, variant in enumerate(self.variants):
            try:
                print(f"\nRunning variant {i+1}/{len(self.variants)}: {variant}")
                metrics = self.run_variant(variant, train_loader, val_loader)
                self.results[variant] = metrics
            except Exception as e:
                print(f"Error running variant {variant}: {str(e)}")
                import traceback
                traceback.print_exc()
                # Continue with next variant
                continue
        
        # Compile results and generate comparison
        self.compile_results()
        
        # Generate cross-variant visualizations
        self.generate_cross_variant_visualizations(val_loader)
    
    def compile_results(self):
        """Compile results from all variants and generate comparison"""
        # Create results table
        results_table = pd.DataFrame(self.results).T
        
        # Save as CSV
        results_table.to_csv(os.path.join(self.ablation_dir, 'all_results.csv'))
        
        # Save as human-readable text file
        with open(os.path.join(self.ablation_dir, 'all_results.txt'), 'w') as f:
            f.write(f"Ablation Study Results: {self.args.ablation_type}\n")
            f.write(f"{'='*50}\n\n")
            
            for i, variant in enumerate(self.variants):
                if variant in self.results:
                    f.write(f"{self.variant_names[i]}\n")
                    f.write(f"{'-'*30}\n")
                    
                    for metric, value in self.results[variant].items():
                        if isinstance(value, float):
                            f.write(f"{metric}: {value:.4f}\n")
                        else:
                            f.write(f"{metric}: {value}\n")
                    
                    f.write("\n")
        
        # Generate visualization
        self.visualize_results()
    
    def visualize_results(self):
        """Generate visualizations comparing the different variants"""
        # Check if we have any results
        if not self.results:
            print("No results to visualize.")
            return
            
        # Plot PSNR comparison
        plt.figure(figsize=(12, 6))
        
        # Get metrics to plot (only if all variants have them)
        metrics_to_plot = []
        for metric in ['psnr', 'ssim']:
            if all(metric in self.results[v] for v in self.results):
                metrics_to_plot.append(metric)
        
        # Create subplots for each metric
        for i, metric in enumerate(metrics_to_plot):
            plt.subplot(1, len(metrics_to_plot), i+1)
            
            # Get data for this metric
            values = []
            labels = []
            
            for v in self.variants:
                if v in self.results and metric in self.results[v]:
                    val = self.results[v][metric]
                    if not np.isnan(val) and np.isfinite(val):
                        values.append(val)
                        labels.append(v)
            
            # Only create bar chart if we have valid values
            if values:
                # Create bar chart
                bars = plt.bar(labels, values)
                
                # Add value labels on top of bars
                for bar in bars:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                            f'{height:.2f}', ha='center', va='bottom', rotation=0)
                
                # Add labels and title
                plt.title(f'{metric.upper()} Comparison')
                plt.ylabel(metric.upper())
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
            else:
                plt.text(0.5, 0.5, f"No valid {metric} values to plot", 
                        ha='center', va='center', transform=plt.gca().transAxes)
        
        # Save figure
        plt.savefig(os.path.join(self.ablation_dir, 'metrics_comparison.png'), dpi=300, bbox_inches='tight')
        
        # If progressive model ablation, plot improvements
        if self.args.ablation_type in ['no_refinement', 'refinement_magnitude']:
            if any('psnr_improvement' in self.results[v] for v in self.results if v in self.results):
                plt.figure(figsize=(12, 6))
                
                # Get variants with improvement metrics
                valid_variants = []
                valid_values = []
                
                for v in self.variants:
                    if v in self.results and 'psnr_improvement' in self.results[v]:
                        val = self.results[v]['psnr_improvement']
                        if not np.isnan(val) and np.isfinite(val):
                            valid_variants.append(v)
                            valid_values.append(val)
                
                # Only create bar chart if we have valid values
                if valid_values:
                    # Plot PSNR improvements
                    bars = plt.bar(valid_variants, valid_values)
                    
                    # Add value labels on top of bars
                    for bar in bars:
                        height = bar.get_height()
                        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                f'{height:.2f}', ha='center', va='bottom', rotation=0)
                    
                    plt.title('PSNR Improvement (Progressive Refinement Stage)')
                    plt.ylabel('PSNR Improvement (dB)')
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    
                    # Save figure
                    plt.savefig(os.path.join(self.ablation_dir, 'refinement_improvement.png'), dpi=300, bbox_inches='tight')
    
    def generate_cross_variant_visualizations(self, val_loader):
        """
        Generate visualizations comparing results across variants
        
        Args:
            val_loader: Validation data loader
        """
        # Create directory
        cross_vis_dir = os.path.join(self.vis_dir, 'cross_variant_comparison')
        create_directory(cross_vis_dir)
        
        # Get a batch of data
        batch = next(iter(val_loader))
        degraded = batch['degraded'].to(self.device)
        reference = batch['reference'].to(self.device)
        filenames = batch['filename']
        
        # Load trained models
        trained_models = {}
        
        for variant in self.variants:
            try:
                # Check if results exist for this variant
                if variant not in self.results:
                    print(f"No results for variant {variant}, skipping visualization")
                    continue
                
                # Create model
                model = self.create_model_for_variant(variant)
                
                # Load checkpoint
                checkpoint_path = os.path.join(self.ablation_dir, variant, 'checkpoints', 'best_model.pth')
                if os.path.exists(checkpoint_path):
                    checkpoint = torch.load(checkpoint_path, map_location=self.device)
                    model.load_state_dict(checkpoint['model'])
                    trained_models[variant] = model
                else:
                    print(f"Could not find checkpoint for variant {variant}")
            except Exception as e:
                print(f"Error loading model for variant {variant}: {str(e)}")
                continue
        
        # Check if we have any trained models
        if not trained_models:
            print("No trained models available for cross-variant visualization.")
            return
        
        # Generate outputs for each model
        all_outputs = {}
        
        for variant, model in trained_models.items():
            model.eval()
            with torch.no_grad():
                try:
                    outputs = model(degraded)
                    
                    # Handle NaN values
                    if torch.isnan(outputs['restored_image']).any():
                        outputs['restored_image'] = torch.nan_to_num(outputs['restored_image'], nan=0.0)
                        outputs['restored_image'] = torch.clamp(outputs['restored_image'], 0.0, 1.0)
                        
                    all_outputs[variant] = outputs
                except Exception as e:
                    print(f"Error generating outputs for variant {variant}: {str(e)}")
                    continue
        
        # Create side-by-side comparison visualizations
        for i in range(min(5, len(degraded))):
            # For each sample, create a grid with results from all variants
            result_images = [degraded[i]]  # Start with degraded image
            result_labels = ['Degraded']
            
            # Add restored images from each variant
            for variant in self.variants:
                if variant in all_outputs:
                    restored = all_outputs[variant]['restored_image'][i]
                    result_images.append(restored)
                    result_labels.append(variant)
            
            # Add reference image
            result_images.append(reference[i])
            result_labels.append('Reference')
            
            try:
                # Create a grid
                grid = torch.stack(result_images)
                grid = vutils.make_grid(grid, nrow=len(result_images), padding=2, normalize=False)
                
                # Save grid
                vutils.save_image(
                    grid,
                    os.path.join(cross_vis_dir, f"comparison_{filenames[i]}")
                )
                
                # Create a labeled composite image with matplotlib
                plt.figure(figsize=(5*len(result_images), 5))
                
                # Convert grid to numpy
                grid_np = grid.cpu().numpy().transpose(1, 2, 0)
                
                # Plot grid
                plt.imshow(grid_np)
                plt.axis('off')
                
                # Add labels
                for j, label in enumerate(result_labels):
                    plt.text(j * grid_np.shape[1] / len(result_labels) + grid_np.shape[1] / (2 * len(result_labels)),
                             grid_np.shape[0] - 20,
                             label,
                             horizontalalignment='center',
                             color='white',
                             fontsize=14,
                             bbox=dict(facecolor='black', alpha=0.5))
                
                # Save figure
                plt.tight_layout()
                plt.savefig(
                    os.path.join(cross_vis_dir, f"labeled_comparison_{filenames[i]}.png"),
                    dpi=300,
                    bbox_inches='tight'
                )
                plt.close()
                
            except Exception as e:
                print(f"Error creating comparison visualization: {e}")
                continue


def run_improved_ablation_study(args):
    """
    Run a complete ablation study with the specified arguments using the improved manager
    
    Args:
        args: Command line arguments
    """
    # Create and run ablation manager
    manager = ImprovedAblationManager(args)
    manager.run_all_variants()
    