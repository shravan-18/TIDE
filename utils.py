import os
import torch
import torch.nn as nn
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt
import torchvision.utils as vutils


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def calculate_psnr(img1, img2):
    """Calculate PSNR between two tensors"""
    # Ensure tensors are in range [0, 1]
    if torch.max(img1) > 1.0 or torch.max(img2) > 1.0:
        img1 = img1 / 255.0 if torch.max(img1) > 1.0 else img1
        img2 = img2 / 255.0 if torch.max(img2) > 1.0 else img2
    
    # Convert to numpy arrays
    img1 = img1.detach().cpu().numpy().transpose(0, 2, 3, 1)  # B, H, W, C
    img2 = img2.detach().cpu().numpy().transpose(0, 2, 3, 1)  # B, H, W, C
    
    # Calculate PSNR for each image in batch
    psnr_values = []
    for i in range(img1.shape[0]):
        psnr_values.append(psnr(img1[i], img2[i], data_range=1.0))
    
    return np.mean(psnr_values)


def calculate_ssim(img1, img2):
    """Calculate SSIM between two tensors"""
    # Ensure tensors are in range [0, 1]
    if torch.max(img1) > 1.0 or torch.max(img2) > 1.0:
        img1 = img1 / 255.0 if torch.max(img1) > 1.0 else img1
        img2 = img2 / 255.0 if torch.max(img2) > 1.0 else img2
    
    # Convert to numpy arrays
    img1 = img1.detach().cpu().numpy().transpose(0, 2, 3, 1)  # B, H, W, C
    img2 = img2.detach().cpu().numpy().transpose(0, 2, 3, 1)  # B, H, W, C
    
    # Calculate SSIM for each image in batch
    ssim_values = []
    for i in range(img1.shape[0]):
        ssim_values.append(ssim(img1[i], img2[i], data_range=1.0, channel_axis=2, multichannel=True))
    
    return np.mean(ssim_values)


def calculate_metrics(restored, reference):
    """Calculate all metrics between restored and reference images"""
    return {
        'psnr': calculate_psnr(restored, reference),
        'ssim': calculate_ssim(restored, reference)
    }


def save_checkpoint(model, optimizer, scheduler, epoch, metrics, best_psnr, path):
    """Save model checkpoint"""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'epoch': epoch,
        'metrics': metrics,
        'best_psnr': best_psnr
    }
    torch.save(checkpoint, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None):
    """Load model checkpoint with proper handling of dynamic layers and optimizer state"""
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    
    # Create dummy input for forward pass to initialize dynamic layers
    dummy_input = torch.zeros(1, 3, 256, 256)
    if torch.cuda.is_available() and next(model.parameters()).is_cuda:
        dummy_input = dummy_input.cuda()
    
    # Run a forward pass with dummy data to initialize dynamic layers
    original_mode = model.training
    model.eval()
    with torch.no_grad():
        _ = model(dummy_input)
    if original_mode:
        model.train()
    
    # Now load the model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Handle optimizer state loading
    if optimizer and 'optimizer_state_dict' in checkpoint:
        # First get the saved optimizer state
        saved_optimizer = checkpoint['optimizer_state_dict']
        
        # Create parameter mapping between saved state and current model
        current_param_mapping = {id(p): p for p in model.parameters() if p.requires_grad}
        
        # Create a new state dict for the current optimizer
        new_state_dict = {
            'state': {},
            'param_groups': saved_optimizer['param_groups']
        }
        
        # Update param_groups to reference current parameters
        for group_idx, group in enumerate(new_state_dict['param_groups']):
            # Map old parameter references to new ones
            param_indices = group['params']
            new_params = []
            
            for old_idx in param_indices:
                # Skip if not found in state
                if old_idx not in saved_optimizer['state']:
                    continue
                
                old_state = saved_optimizer['state'][old_idx]
                old_param_shape = old_state.get('exp_avg', next(iter(old_state.values()))).shape
                
                # Find a matching parameter in our current model
                found = False
                for p in model.parameters():
                    if p.requires_grad and p.shape == old_param_shape:
                        new_params.append(id(p))
                        # Copy state from old param to new param
                        new_state_dict['state'][id(p)] = saved_optimizer['state'][old_idx]
                        found = True
                        break
                
                if not found:
                    # Skip this parameter since no match was found
                    pass
            
            # Update the group with new parameter references
            group['params'] = new_params
        
        # Load the modified state dict into optimizer
        try:
            optimizer.load_state_dict(new_state_dict)
            print("Optimizer state loaded successfully with parameter remapping")
        except Exception as e:
            print(f"Error loading optimizer state: {e}")
            print("Continuing with fresh optimizer state")
    
    # Load scheduler state if available
    if scheduler and 'scheduler_state_dict' in checkpoint:
        try:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("Scheduler state loaded successfully")
        except Exception as e:
            print(f"Could not load scheduler state: {e}")
            print("Continuing with fresh scheduler state")
    
    return checkpoint


def visualize_results(degraded, restored, reference, degradation_maps=None, hypotheses=None, 
                      initial_restoration=None, refinement=None, residual_maps=None, save_path=None):
    """Visualize restoration results with support for progressive refinement"""
    
    if initial_restoration is not None and refinement is not None:
        # Progressive model visualization
        fig = plt.figure(figsize=(20, 10))
        
        # Determine number of rows and columns
        num_images = 5  # degraded, initial, refinement, final, reference
        if hypotheses is not None:
            num_images += len(hypotheses)
        
        # Create subplots
        cols = num_images
        rows = 1
        
        # Plot degraded image
        plt.subplot(rows, cols, 1)
        plt.imshow(degraded)
        plt.title('Degraded')
        plt.axis('off')
        
        # Plot initial restoration J_hat_1
        plt.subplot(rows, cols, 2)
        plt.imshow(initial_restoration)
        plt.title('Initial Restoration')
        plt.axis('off')
        
        # Plot refinement (amplified for visibility)
        plt.subplot(rows, cols, 3)
        # Amplify refinement to make it visible
        visible_refinement = refinement * 5 + 0.5
        visible_refinement = np.clip(visible_refinement, 0, 1)
        plt.imshow(visible_refinement)
        plt.title('Refinement (5x)')
        plt.axis('off')
        
        # Plot final restored image
        plt.subplot(rows, cols, 4)
        plt.imshow(restored)
        plt.title('Final Restored')
        plt.axis('off')
        
        # Plot reference image
        plt.subplot(rows, cols, 5)
        plt.imshow(reference)
        plt.title('Reference')
        plt.axis('off')
        
        # Plot hypotheses if available
        if hypotheses is not None:
            for i, hyp in enumerate(hypotheses):
                plt.subplot(rows, cols, 6 + i)
                plt.imshow(hyp)
                plt.title(f'Hypothesis {i+1}')
                plt.axis('off')
        
        # Create a separate figure for degradation maps
        if degradation_maps is not None or residual_maps is not None:
            fig2 = plt.figure(figsize=(20, 5))
            map_cols = 0
            
            # Count number of map types
            if degradation_maps is not None:
                map_cols += degradation_maps.shape[0]
            if residual_maps is not None:
                map_cols += residual_maps.shape[0]
            
            col_idx = 1
            
            # Plot degradation maps
            if degradation_maps is not None:
                for i in range(degradation_maps.shape[0]):
                    plt.subplot(1, map_cols, col_idx)
                    plt.imshow(degradation_maps[i], cmap='viridis')
                    plt.title(f'Degradation Map {i+1}')
                    plt.axis('off')
                    col_idx += 1
            
            # Plot residual degradation maps Mr maps
            if residual_maps is not None:
                for i in range(residual_maps.shape[0]):
                    plt.subplot(1, map_cols, col_idx)
                    plt.imshow(residual_maps[i], cmap='plasma')
                    plt.title(f'Residual Map {i+1}')
                    plt.axis('off')
                    col_idx += 1
            
            # Save or show degradation maps
            if save_path:
                maps_path = save_path.replace('.png', '_maps.png')
                plt.savefig(maps_path, bbox_inches='tight')
                plt.close(fig2)
            
    else:
        # Standard model visualization
        fig = plt.figure(figsize=(15, 10))
        
        # Determine number of rows and columns
        num_images = 3  # degraded, restored, reference
        if hypotheses is not None:
            num_images += len(hypotheses)
        
        # Create subplots
        rows = 1
        cols = num_images
        
        # Plot degraded image
        plt.subplot(rows, cols, 1)
        plt.imshow(degraded)
        plt.title('Degraded')
        plt.axis('off')
        
        # Plot hypotheses if available
        if hypotheses is not None:
            for i, hyp in enumerate(hypotheses):
                plt.subplot(rows, cols, 2 + i)
                plt.imshow(hyp)
                plt.title(f'Hypothesis {i+1}')
                plt.axis('off')
        
        # Plot restored image
        plt.subplot(rows, cols, cols - 1)
        plt.imshow(restored)
        plt.title('Restored')
        plt.axis('off')
        
        # Plot reference image
        plt.subplot(rows, cols, cols)
        plt.imshow(reference)
        plt.title('Reference')
        plt.axis('off')
        
        # Plot degradation maps if available
        if degradation_maps is not None:
            fig2 = plt.figure(figsize=(15, 5))
            num_maps = degradation_maps.shape[0]
            for i in range(num_maps):
                plt.subplot(1, num_maps, i + 1)
                plt.imshow(degradation_maps[i], cmap='viridis')
                plt.title(f'Degradation Map {i+1}')
                plt.axis('off')
            
            # Save or show degradation maps
            if save_path:
                maps_path = save_path.replace('.png', '_maps.png')
                plt.savefig(maps_path, bbox_inches='tight')
                plt.close(fig2)
    
    # Save or show
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.tight_layout()
        plt.show()


def tensor_to_image(tensor):
    """Convert a torch tensor to a numpy image"""
    if len(tensor.shape) == 4:  # Batch of images
        tensor = tensor[0]  # Take first image
    
    img = tensor.detach().cpu().numpy()
    img = np.transpose(img, (1, 2, 0))  # C, H, W -> H, W, C
    
    # Ensure range [0, 1]
    img = np.clip(img, 0, 1)
    
    return img


def make_grid(tensors, nrow=8, padding=2, normalize=True):
    """Make a grid of images from a list of tensors"""
    return vutils.make_grid(tensors, nrow=nrow, padding=padding, normalize=normalize)


def save_image(tensor, filename, nrow=8, padding=2, normalize=True):
    """Save a tensor as an image"""
    vutils.save_image(tensor, filename, nrow=nrow, padding=padding, normalize=normalize)


def create_directory(directory):
    """Create a directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")
    return directory


def count_parameters(model):
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_epoch_images(degraded, restored, reference, hypotheses=None, degradation_maps=None,
                      initial_restoration=None, refinement=None, residual_maps=None,
                      save_dir=None, epoch=0, phase='train', max_images=10):
    """
    Save a sample of images after each epoch for monitoring with support for progressive refinement.
    
    Args:
        degraded (torch.Tensor): Batch of degraded input images [B, C, H, W]
        restored (torch.Tensor): Batch of restored output images [B, C, H, W]
        reference (torch.Tensor): Batch of ground truth reference images [B, C, H, W]
        hypotheses (list of torch.Tensor, optional): List of hypothesis tensors
        degradation_maps (torch.Tensor, optional): Degradation maps [B, D, H, W]
        initial_restoration (torch.Tensor, optional): Initial restoration from base model [B, C, H, W]
        refinement (torch.Tensor, optional): Refinement correction term [B, C, H, W]
        residual_maps (torch.Tensor, optional): Residual degradation maps [B, D, H, W]
        save_dir (str): Directory to save images
        epoch (int): Current epoch number
        phase (str): 'train' or 'val'
        max_images (int): Maximum number of images to save (default: 10)
    """
    # Create directory for this epoch and phase
    image_dir = os.path.join(save_dir, 'images', f'epoch_{epoch}', phase)
    create_directory(image_dir)
    
    # Limit number of images to save
    n_images = min(degraded.size(0), max_images)
    
    # Ensure tensors are in the right range [0, 1]
    degraded = torch.clamp(degraded, 0, 1)
    restored = torch.clamp(restored, 0, 1)
    reference = torch.clamp(reference, 0, 1)
    
    # Check if this is a progressive refinement model
    is_progressive = initial_restoration is not None and refinement is not None
    
    # Create and save grid of all sample images
    all_images = []
    
    for i in range(n_images):
        if is_progressive:
            # For progressive model: degraded, initial, refinement, final, reference
            initial_vis = initial_restoration[i]
            refinement_vis = refinement[i]
            
            # Scale refinement for visibility (it's usually small)
            refinement_vis_scaled = refinement_vis * 5 + 0.5
            refinement_vis_scaled = torch.clamp(refinement_vis_scaled, 0, 1)
            
            row = [degraded[i], initial_vis, refinement_vis_scaled, restored[i], reference[i]]
            
            # Save before-after comparison separately
            compare_grid = make_grid([degraded[i], initial_vis, restored[i], reference[i]], 
                                     nrow=4, padding=2, normalize=False)
            save_image(compare_grid, os.path.join(image_dir, f'compare_{i}.png'))
            
            # Save close-up regions to show the refinement effect
            # This would require more complex region selection logic
            # Could be implemented in a future enhancement
        else:
            # For standard model: degraded, hypotheses, restored, reference
            row = [degraded[i]]
            
            # Add hypotheses if available
            if hypotheses:
                for h in hypotheses:
                    row.append(h[i])
            
            # Add restored and reference
            row.append(restored[i])
            row.append(reference[i])
        
        all_images.extend(row)
    
    # Calculate number of columns for the grid
    if is_progressive:
        n_cols = 5  # degraded, initial, refinement, restored, reference
    else:
        n_cols = 3  # degraded, restored, reference
        if hypotheses:
            n_cols += len(hypotheses)  # add hypotheses
    
    # Save grid of images
    grid = make_grid(all_images, nrow=n_cols, padding=2, normalize=False)
    save_image(grid, os.path.join(image_dir, f'samples_grid.png'))
    
    # Save degradation maps if available
    if degradation_maps is not None:
        deg_images = []
        for i in range(n_images):
            for j in range(degradation_maps.size(1)):
                # Make single-channel map into 3-channel
                deg_map = degradation_maps[i, j].unsqueeze(0).repeat(3, 1, 1)
                deg_images.append(deg_map)
        
        # Save grid of degradation maps
        deg_grid = make_grid(deg_images, nrow=degradation_maps.size(1), padding=2, normalize=True)
        save_image(deg_grid, os.path.join(image_dir, f'degradation_maps.png'))
    
    # Save residual degradation maps Mr maps if available
    if residual_maps is not None:
        res_images = []
        for i in range(n_images):
            for j in range(residual_maps.size(1)):
                # Make single-channel map into 3-channel
                res_map = residual_maps[i, j].unsqueeze(0).repeat(3, 1, 1)
                res_images.append(res_map)
        
        # Save grid of residual maps
        res_grid = make_grid(res_images, nrow=residual_maps.size(1), padding=2, normalize=True)
        save_image(res_grid, os.path.join(image_dir, f'residual_maps.png'))
    
    print(f"Saved {n_images} sample images for epoch {epoch} ({phase}) to {image_dir}")


def compare_progressive_improvement(degraded, initial_restoration, final_restoration, reference, 
                                   save_path=None, metrics=None):
    """Create a visualization comparing initial and final restoration J_hats with metrics"""
    # Create figure with 4 subplots: degraded, initial, final, reference
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Convert tensors to numpy images if needed
    if isinstance(degraded, torch.Tensor):
        degraded = tensor_to_image(degraded)
    if isinstance(initial_restoration, torch.Tensor):
        initial_restoration = tensor_to_image(initial_restoration)
    if isinstance(final_restoration, torch.Tensor):
        final_restoration = tensor_to_image(final_restoration)
    if isinstance(reference, torch.Tensor):
        reference = tensor_to_image(reference)
    
    # Plot the images
    axes[0].imshow(degraded)
    axes[0].set_title('Degraded')
    axes[0].axis('off')
    
    axes[1].imshow(initial_restoration)
    if metrics and 'psnr_initial' in metrics and 'ssim_initial' in metrics:
        axes[1].set_title(f'Initial: PSNR={metrics["psnr_initial"]:.2f}, SSIM={metrics["ssim_initial"]:.4f}')
    else:
        axes[1].set_title('Initial Restoration')
    axes[1].axis('off')
    
    axes[2].imshow(final_restoration)
    if metrics and 'psnr' in metrics and 'ssim' in metrics:
        axes[2].set_title(f'Final: PSNR={metrics["psnr"]:.2f}, SSIM={metrics["ssim"]:.4f}')
    else:
        axes[2].set_title('Final Restoration')
    axes[2].axis('off')
    
    axes[3].imshow(reference)
    axes[3].set_title('Reference')
    axes[3].axis('off')
    
    # Add improvement annotation
    if metrics and 'psnr_improvement' in metrics and 'ssim_improvement' in metrics:
        fig.suptitle(f'Progressive Improvement: PSNR +{metrics["psnr_improvement"]:.2f}dB, SSIM +{metrics["ssim_improvement"]:.4f}', 
                    fontsize=16)
    
    # Save or show
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.tight_layout()
        plt.show()
    
    return fig
