import argparse
import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from model import DGMHRN, TIDE
from utils import create_directory, load_checkpoint
# Import the BRISQUE module
from brisque import BRISQUE

# Set matplotlib parameters for high-quality output
matplotlib.rcParams['figure.figsize'] = (10, 8)
matplotlib.rcParams['figure.dpi'] = 150
matplotlib.rcParams['savefig.dpi'] = 200
matplotlib.rcParams['font.size'] = 11
matplotlib.rcParams['axes.titlesize'] = 12
matplotlib.rcParams['axes.titleweight'] = 'bold'
matplotlib.rcParams['figure.titlesize'] = 14
matplotlib.rcParams['figure.titleweight'] = 'bold'

# Initialize BRISQUE calculator
brisque_calculator = BRISQUE(url=False)

# Unsupervised metrics implementation
def calculate_uiqm(img_np):
    """
    Calculate Underwater Image Quality Measure (UIQM)
    img_np: numpy array of shape (H, W, 3) in RGB format, values in [0, 1]
    """
    # Convert to BGR format and scale to [0, 255]
    img_bgr = (img_np[:, :, ::-1] * 255).astype(np.uint8)
    
    # Calculate colorfulness (UICM)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Calculate mean and standard deviation for a* and b* channels
    mu_a = np.mean(a)
    mu_b = np.mean(b)
    std_a = np.std(a)
    std_b = np.std(b)
    
    # UICM formula
    uicm = -0.0268 * np.sqrt(mu_a**2 + mu_b**2) + 0.1586 * np.sqrt(std_a**2 + std_b**2)
    
    # Calculate sharpness (UISM)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobelx**2 + sobely**2)
    uism = np.mean(sobel_mag)
    
    # Calculate contrast (UIConM)
    uiconm = np.std(gray) / np.mean(gray) if np.mean(gray) > 0 else 0
    
    # UIQM combines these three components
    c1, c2, c3 = 0.0282, 0.2953, 3.5753  # Weights from the paper
    uiqm = c1 * uicm + c2 * uism + c3 * uiconm
    
    return uiqm

def calculate_uism(img_np):
    """
    Calculate Underwater Image Sharpness Measure (UISM)
    img_np: numpy array of shape (H, W, 3) in RGB format, values in [0, 1]
    """
    # Convert to BGR and scale to [0, 255]
    img_bgr = (img_np[:, :, ::-1] * 255).astype(np.uint8)
    
    # Split into color channels
    b, g, r = cv2.split(img_bgr)
    
    # Calculate edge map for each channel
    edge_r = cv2.Sobel(r, cv2.CV_64F, 1, 1, ksize=3)
    edge_g = cv2.Sobel(g, cv2.CV_64F, 1, 1, ksize=3)
    edge_b = cv2.Sobel(b, cv2.CV_64F, 1, 1, ksize=3)
    
    # Calculate sharpness for each channel
    sharp_r = np.mean(np.abs(edge_r))
    sharp_g = np.mean(np.abs(edge_g))
    sharp_b = np.mean(np.abs(edge_b))
    
    # UISM is weighted sum (green channel gets highest weight in underwater images)
    weights = [0.299, 0.587, 0.114]  # RGB weights
    uism = weights[0] * sharp_r + weights[1] * sharp_g + weights[2] * sharp_b
    
    return uism

def calculate_brisque(img_np):
    """
    Calculate BRISQUE (Blind/Referenceless Image Spatial Quality Evaluator)
    img_np: numpy array of shape (H, W, 3) in RGB format, values in [0, 1]
    
    Uses the PyPI brisque module
    """
    try:
        # Convert image to uint8 (0-255 range) if it's in float format
        if img_np.dtype != np.uint8:
            img_uint8 = (img_np * 255).astype(np.uint8)
        else:
            img_uint8 = img_np
        
        # Calculate BRISQUE score
        score = brisque_calculator.score(img_uint8)
        return score
    except Exception as e:
        print(f"Warning: BRISQUE calculation failed: {e}")
        # Return a placeholder value on error
        return 100.0

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate underwater image restoration model on a single image')
    
    # Input/output arguments
    parser.add_argument('--input_path', type=str, required=True, help='Path to input underwater image')
    parser.add_argument('--output_dir', type=str, default='single_eval_results', help='Directory to save results')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    
    # Model arguments
    parser.add_argument('--progressive_checkpoint', type=str, required=True, help='Path to progressive model checkpoint')
    parser.add_argument('--base_channels', type=int, default=64, help='Base number of channels in the model')
    parser.add_argument('--num_downs', type=int, default=5, help='Number of downsampling layers in encoder')
    parser.add_argument('--num_degradation_types', type=int, default=4, help='Number of degradation types (color, contrast, detail, noise) to model')
    parser.add_argument('--norm_type', type=str, default='instance', help='Normalization type')
    parser.add_argument('--activation', type=str, default='leaky_relu', help='Activation function')
    parser.add_argument('--fusion_type', type=str, default='learned', help='Fusion mechanism type')
    parser.add_argument('--img_size', type=int, default=256, help='Input image size')
    
    # Output options
    parser.add_argument('--save_hypotheses', action='store_true', help='Save individual hypotheses')
    parser.add_argument('--save_degradation_maps', action='store_true', help='Save degradation maps')
    parser.add_argument('--save_refinement', action='store_true', help='Save refinement maps and residual degradation maps Mr maps')
    parser.add_argument('--visualize', action='store_true', help='Create visualization figures')
    
    args = parser.parse_args()
    return args

def load_model(args, device):
    """Load the progressive model from checkpoint"""
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
    
    # Create progressive model
    model = TIDE(
        base_model=base_model,
        num_degradation_types=args.num_degradation_types,
        norm_type=args.norm_type,
        activation=args.activation
    ).to(device)
    
    # Load checkpoint
    checkpoint = load_checkpoint(args.progressive_checkpoint, model)
    print(f"Loaded progressive model from: {args.progressive_checkpoint}")
    
    return model

def preprocess_image(image_path, img_size, device):
    """Load and preprocess an image for model input"""
    img = Image.open(image_path).convert('RGB')
    original_img = img.copy()
    
    # Resize image if needed
    if img_size:
        img = img.resize((img_size, img_size), Image.LANCZOS)
    
    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Apply transformations
    img_tensor = transform(img).unsqueeze(0).to(device)  # Add batch dimension
    
    return img_tensor, original_img

def save_image(tensor, path, original_size=None):
    """Save a tensor as an image"""
    if tensor is None:
        print(f"Warning: Cannot save None tensor to {path}")
        return None
        
    # Convert tensor to PIL Image (assuming tensor is in range [0, 1])
    tensor = torch.clamp(tensor, 0, 1)  # Ensure values are in valid range
    img = transforms.ToPILImage()(tensor.cpu().squeeze(0))
    
    # Resize back to original size if provided
    if original_size:
        img = img.resize((original_size[0], original_size[1]), Image.LANCZOS)
    
    # Save image
    img.save(path)
    print(f"Saved image to {path}")
    return img

def create_overlay(img_np, map_np, colormap='inferno', alpha=0.6):
    """
    Create an overlay of a heatmap on an image
    
    Args:
        img_np: Base image as numpy array (H, W, 3) with values in [0, 1]
        map_np: Map to overlay as numpy array (H, W) with values in [0, 1]
        colormap: Matplotlib colormap name
        alpha: Transparency of the overlay (0-1)
        
    Returns:
        Overlay image as numpy array (H, W, 3) with values in [0, 1]
    """
    # Convert to uint8 for visualization
    img_uint8 = (img_np * 255).astype(np.uint8)
    
    # Apply colormap to the degradation map
    cmap = plt.get_cmap(colormap)
    colored_map = cmap(map_np)
    colored_map_uint8 = (colored_map[:, :, :3] * 255).astype(np.uint8)
    
    # Create overlay
    overlay = cv2.addWeighted(img_uint8, 1-alpha, colored_map_uint8, alpha, 0)
    
    # Convert back to float [0, 1]
    return overlay.astype(np.float32) / 255.0

def visualize_degradation_maps_with_overlay(image_np, maps, output_dir, prefix="degradation_map", 
                                           original_size=None, cmaps=['inferno', 'viridis', 'plasma', 'magma']):
    """
    Visualize degradation maps as heatmaps and overlay them on the original image
    
    Args:
        image_np: Original image as numpy array (H, W, 3) with values in [0, 1]
        maps: Degradation maps as tensor (C, H, W) with values in [0, 1]
        output_dir: Directory to save visualizations
        prefix: Prefix for filenames
        original_size: Tuple (width, height) for resizing maps to original image size
        cmaps: List of colormaps to use for different maps
    """
    if maps is None or maps.dim() == 0:
        print(f"Warning: Cannot visualize empty degradation maps")
        return
    
    # Ensure directory exists
    maps_dir = os.path.join(output_dir, "maps")
    os.makedirs(maps_dir, exist_ok=True)
    overlays_dir = os.path.join(output_dir, "overlays")
    os.makedirs(overlays_dir, exist_ok=True)
    
    # Process each map
    maps_np = maps.cpu().numpy()
    num_maps = maps_np.shape[0]
    
    # Resize maps to original image size if provided
    if original_size:
        # Get original dimensions
        orig_width, orig_height = original_size
        orig_image = cv2.resize(image_np, (orig_width, orig_height), interpolation=cv2.INTER_LANCZOS4)
        
        # Resize maps
        resized_maps = []
        for i in range(num_maps):
            map_resized = cv2.resize(maps_np[i], (orig_width, orig_height), interpolation=cv2.INTER_LANCZOS4)
            resized_maps.append(map_resized)
        maps_np = np.array(resized_maps)
    else:
        orig_image = image_np
    
    # Create a figure for all maps and overlays
    fig, axes = plt.subplots(2, num_maps, figsize=(num_maps * 4, 8))
    if num_maps == 1:
        axes = np.array([[axes[0]], [axes[1]]])
    
    # Map titles based on prefix
    if "degradation" in prefix.lower():
        map_titles = ["Color Degradation", "Contrast Loss", "Detail Loss", "Noise Degradation"]
    elif "residual" in prefix.lower():
        map_titles = ["Residual Color", "Residual Contrast", "Residual Detail", "Residual Noise"]
    else:
        map_titles = [f"Map {i+1}" for i in range(num_maps)]
    
    # Process each map
    for i in range(num_maps):
        # Get colormap (cycle through provided list)
        cmap_name = cmaps[i % len(cmaps)]
        cmap = plt.get_cmap(cmap_name)
        
        # Create heatmap visualization
        axes[0, i].imshow(maps_np[i], cmap=cmap)
        axes[0, i].set_title(f"{map_titles[i]}", fontweight='bold')
        axes[0, i].axis('off')
        
        # Create overlay
        overlay = create_overlay(orig_image, maps_np[i], colormap=cmap_name, alpha=0.5)
        axes[1, i].imshow(overlay)
        axes[1, i].set_title(f"{map_titles[i]} Overlay", fontweight='bold')
        axes[1, i].axis('off')
        
        # Save individual visualizations
        plt.figure(figsize=(6, 6))
        plt.imshow(maps_np[i], cmap=cmap)
        plt.title(f"{map_titles[i]}", fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(maps_dir, f"{prefix}_{i+1}.png"))
        plt.close()
        
        # Save individual overlays
        plt.figure(figsize=(6, 6))
        plt.imshow(overlay)
        plt.title(f"{map_titles[i]} Overlay", fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(overlays_dir, f"{prefix}_overlay_{i+1}.png"))
        plt.close()
    
    # Add super title
    title_name = "Degradation Maps" if "degradation" in prefix.lower() else "Residual Degradation Maps"
    fig.suptitle(f"{title_name} and Overlays", fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle
    
    # Save combined figure
    plt.savefig(os.path.join(output_dir, f"{prefix}_combined.png"))
    plt.close()
    
    print(f"Saved {num_maps} {prefix} visualizations with overlays")

def create_image_grid(image_tensors, num_cols, titles=None, original_size=None):
    """Create a grid of images from tensors with optional titles"""
    num_images = len(image_tensors)
    num_rows = (num_images + num_cols - 1) // num_cols
    
    # Convert tensors to PIL images
    pil_images = []
    for tensor in image_tensors:
        tensor = torch.clamp(tensor, 0, 1)
        img = transforms.ToPILImage()(tensor.cpu().squeeze(0))
        if original_size:
            img = img.resize((original_size[0], original_size[1]), Image.LANCZOS)
        pil_images.append(img)
    
    # Get dimensions
    width = pil_images[0].width
    height = pil_images[0].height
    
    # Create figure for grid with titles
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 4, num_rows * 4))
    if num_rows == 1 and num_cols == 1:
        axes = np.array([[axes]])
    elif num_rows == 1:
        axes = axes.reshape(1, -1)
    elif num_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Add images and titles
    for i, img in enumerate(pil_images):
        if i < num_images:
            row = i // num_cols
            col = i % num_cols
            axes[row, col].imshow(np.array(img))
            if titles and i < len(titles):
                axes[row, col].set_title(titles[i], fontweight='bold')
            axes[row, col].axis('off')
        else:
            # Hide unused subplots
            row = i // num_cols
            col = i % num_cols
            axes[row, col].axis('off')
    
    plt.tight_layout()
    return fig

def visualize_results(input_img, restored_img, output_dir, initial_img=None, 
                     metrics=None, refinement=None, hypotheses=None):
    """Create comprehensive visualization of restoration results"""
    # Convert PIL images to numpy arrays
    input_np = np.array(input_img).astype(np.float32) / 255.0
    restored_np = np.array(restored_img).astype(np.float32) / 255.0
    
    # 1. Create basic comparison visualization
    fig, axes = plt.subplots(1, 2 if initial_img is None else 3, figsize=(12, 5))
    
    # Input image with metrics
    axes[0].imshow(input_np)
    title = "Input Underwater Image"
    if metrics:
        title += f"\nUIQM: {metrics['input_uiqm']:.2f}, UISM: {metrics['input_uism']:.2f}, BRISQUE: {metrics['input_brisque']:.2f}"
    axes[0].set_title(title, fontweight='bold')
    axes[0].axis('off')
    
    # If we have intermediate restoration
    if initial_img is not None:
        initial_np = np.array(initial_img).astype(np.float32) / 255.0
        axes[1].imshow(initial_np)
        axes[1].set_title("Initial Restoration\n(Base Model)", fontweight='bold')
        axes[1].axis('off')
        
        # Final restoration
        axes[2].imshow(restored_np)
        title = "Final Restoration\n(After Refinement)"
        if metrics:
            title += f"\nUIQM: {metrics['restored_uiqm']:.2f}, UISM: {metrics['restored_uism']:.2f}, BRISQUE: {metrics['restored_brisque']:.2f}"
        axes[2].set_title(title, fontweight='bold')
        axes[2].axis('off')
    else:
        # Just final restoration J_hat
        axes[1].imshow(restored_np)
        title = "Restored Image"
        if metrics:
            title += f"\nUIQM: {metrics['restored_uiqm']:.2f}, UISM: {metrics['restored_uism']:.2f}, BRISQUE: {metrics['restored_brisque']:.2f}"
        axes[1].set_title(title, fontweight='bold')
        axes[1].axis('off')
    
    # Add improvement text if metrics are available
    if metrics and 'uiqm_improvement' in metrics:
        fig.suptitle("Underwater Image Restoration\n" + 
                    f"Improvements: UIQM: {'+' if metrics['uiqm_improvement'] >= 0 else ''}{metrics['uiqm_improvement']:.2f}, " +
                    f"UISM: {'+' if metrics['uism_improvement'] >= 0 else ''}{metrics['uism_improvement']:.2f}, " +
                    f"BRISQUE: {'+' if metrics['brisque_improvement'] >= 0 else ''}{metrics['brisque_improvement']:.2f}",
                    fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.9])  # Adjust for suptitle
    else:
        plt.tight_layout()
    
    # Save the comparison
    comparison_path = os.path.join(output_dir, 'comparison.png')
    plt.savefig(comparison_path)
    plt.close(fig)
    print(f"Saved comparison to {comparison_path}")
    
    # 2. Create refinement visualization if available
    if refinement is not None and initial_img is not None:
        refinement_np = np.array(refinement).astype(np.float32) / 255.0
        
        # Create a visualization specifically for the refinement process
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        # Input image
        axes[0].imshow(input_np)
        axes[0].set_title("Input Image", fontweight='bold')
        axes[0].axis('off')
        
        # Initial restoration
        axes[1].imshow(initial_np)
        axes[1].set_title("Initial Restoration", fontweight='bold')
        axes[1].axis('off')
        
        # Refinement map (amplified for visibility)
        axes[2].imshow(refinement_np)
        axes[2].set_title("Refinement\n(5x amplified)", fontweight='bold')
        axes[2].axis('off')
        
        # Final restored image
        axes[3].imshow(restored_np)
        axes[3].set_title("Final Restoration", fontweight='bold')
        axes[3].axis('off')
        
        plt.tight_layout()
        refinement_path = os.path.join(output_dir, 'refinement_process.png')
        plt.savefig(refinement_path)
        plt.close(fig)
        print(f"Saved refinement process visualization to {refinement_path}")
    
    # 3. Create hypothesis comparison if available
    if hypotheses and len(hypotheses) > 0:
        # Convert hypothesis tensors to numpy arrays
        original_size = input_img.size  # (width, height)
        hypotheses_np = [np.array(save_image(h, os.path.join(output_dir, f'hypothesis_{i}.png'), original_size))
                         .astype(np.float32) / 255.0 for i, h in enumerate(hypotheses)]
        
        # Create a visualization of all hypotheses
        num_hypotheses = len(hypotheses_np)
        fig, axes = plt.subplots(2, num_hypotheses + 1, figsize=((num_hypotheses + 1) * 4, 8))
        
        # First row: Hypotheses
        hypothesis_titles = ["Color Restoration", "Contrast Enhancement", 
                             "Detail Recovery", "Denoising"]
        
        for i, hyp_np in enumerate(hypotheses_np):
            axes[0, i].imshow(hyp_np)
            axes[0, i].set_title(f"Hypothesis {i+1}:\n{hypothesis_titles[i]}", fontweight='bold')
            axes[0, i].axis('off')
        
        # First row, last column: Input image
        axes[0, -1].imshow(input_np)
        axes[0, -1].set_title("Input Image", fontweight='bold')
        axes[0, -1].axis('off')
        
        # Second row: First column is final fusion
        axes[1, 0].imshow(restored_np)
        axes[1, 0].set_title("Final Fused Result", fontweight='bold')
        axes[1, 0].axis('off')
        
        # Second row: Rest are empty or could be used for zoomed regions
        for i in range(1, num_hypotheses + 1):
            axes[1, i].axis('off')
        
        # Add title
        fig.suptitle("Multi-Hypothesis Restoration Process", fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        hypotheses_path = os.path.join(output_dir, 'hypotheses_comparison.png')
        plt.savefig(hypotheses_path)
        plt.close(fig)
        print(f"Saved hypotheses comparison to {hypotheses_path}")
    
    # 4. If we have both hypotheses and refinement, create a full pipeline visualization
    if hypotheses and len(hypotheses) > 0 and refinement is not None and initial_img is not None:
        # Create a flowchart-style visualization
        fig = plt.figure(figsize=(15, 10))
        
        # Define grid for complex layout
        gs = fig.add_gridspec(3, 3)
        
        # Input and final output (larger)
        ax_input = fig.add_subplot(gs[0, 0])
        ax_input.imshow(input_np)
        ax_input.set_title("Input Image", fontweight='bold')
        ax_input.axis('off')
        
        ax_output = fig.add_subplot(gs[0, 2])
        ax_output.imshow(restored_np)
        ax_output.set_title("Final Restoration", fontweight='bold')
        ax_output.axis('off')
        
        # Initial restoration
        ax_initial = fig.add_subplot(gs[0, 1])
        ax_initial.imshow(initial_np)
        ax_initial.set_title("Initial Restoration", fontweight='bold')
        ax_initial.axis('off')
        
        # Hypotheses row
        ax_hyp = []
        for i in range(min(4, len(hypotheses_np))):
            ax = fig.add_subplot(gs[1, i if i < 3 else 2])
            ax.imshow(hypotheses_np[i])
            ax.set_title(f"{hypothesis_titles[i]}", fontweight='bold')
            ax.axis('off')
            ax_hyp.append(ax)
        
        # Refinement visualization
        ax_refinement = fig.add_subplot(gs[2, 1])
        ax_refinement.imshow(refinement_np)
        ax_refinement.set_title("Refinement\n(5x amplified)", fontweight='bold')
        ax_refinement.axis('off')
        
        # Add annotations with arrows to show the flow
        plt.figtext(0.5, 0.95, "DGMHRN Progressive Underwater Image Restoration Pipeline", 
                   ha="center", fontsize=16, fontweight='bold')
        
        # Add some explanatory text
        plt.figtext(0.1, 0.05, "Stage 1: Initial restoration with multi-hypothesis approach", 
                   fontsize=10)
        plt.figtext(0.6, 0.05, "Stage 2: Progressive refinement", 
                   fontsize=10)
        
        # Save the full pipeline visualization
        plt.tight_layout(rect=[0, 0.07, 1, 0.93])
        pipeline_path = os.path.join(output_dir, 'full_pipeline.png')
        plt.savefig(pipeline_path)
        plt.close(fig)
        print(f"Saved full pipeline visualization to {pipeline_path}")

def compare_progressive_improvement(input_img, initial_img, restored_img, output_dir, metrics=None):
    """Create a detailed visualization comparing initial and final restoration J_hats with metrics"""
    # Convert images to numpy
    input_np = np.array(input_img).astype(np.float32) / 255.0
    initial_np = np.array(initial_img).astype(np.float32) / 255.0
    restored_np = np.array(restored_img).astype(np.float32) / 255.0
    
    # Create figure with detailed comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Input image
    axes[0].imshow(input_np)
    axes[0].set_title("Input Underwater Image", fontweight='bold')
    axes[0].axis('off')
    
    # Initial restoration with metrics
    axes[1].imshow(initial_np)
    title = "Initial Restoration (Base Model)"
    if metrics:
        title += f"\nUIQM: {metrics['input_uiqm']:.2f}"
    axes[1].set_title(title, fontweight='bold')
    axes[1].axis('off')
    
    # Final restoration with metrics
    axes[2].imshow(restored_np)
    title = "Final Restoration (After Refinement)"
    if metrics:
        title += f"\nUIQM: {metrics['restored_uiqm']:.2f}"
    axes[2].set_title(title, fontweight='bold')
    axes[2].axis('off')
    
    # Add improvement text if metrics are available
    if metrics and 'uiqm_improvement' in metrics:
        fig.suptitle("Progressive Underwater Image Restoration\n" + 
                    f"Quality Improvement: UIQM: {'+' if metrics['uiqm_improvement'] >= 0 else ''}{metrics['uiqm_improvement']:.2f}, " +
                    f"UISM: {'+' if metrics['uism_improvement'] >= 0 else ''}{metrics['uism_improvement']:.2f}, " +
                    f"BRISQUE: {'+' if metrics['brisque_improvement'] >= 0 else ''}{metrics['brisque_improvement']:.2f}",
                    fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.9])  # Adjust for suptitle
    else:
        plt.tight_layout()
    
    # Save comparison
    comparison_path = os.path.join(output_dir, 'progressive_improvement.png')
    plt.savefig(comparison_path)
    plt.close(fig)
    print(f"Saved progressive improvement comparison to {comparison_path}")

def main():
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    create_directory(args.output_dir)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(args, device)
    model.eval()
    
    # Load and preprocess input image
    input_tensor, original_img = preprocess_image(args.input_path, args.img_size, device)
    original_size = original_img.size  # (width, height)
    
    # Run inference
    print("Running inference...")
    with torch.no_grad():
        # Forward pass through the model
        outputs = model(input_tensor)
        
        # Print output keys for debugging
        print(f"Model output keys: {outputs.keys()}")
        
        # Extract the restored image (final output)
        restored_tensor = outputs['restored_image']
        
        # Extract other outputs if available
        initial_restoration = outputs.get('initial_restoration', None)
        refinement = outputs.get('refinement', None)
        degradation_maps = outputs.get('degradation_maps', None)
        residual_maps = outputs.get('residual_maps', None)
        hypotheses = outputs.get('hypotheses', [])
        refinements = outputs.get('refinements', [])
    
    # Save results
    print("Saving results...")
    # Save input image
    input_path = os.path.join(args.output_dir, 'input.png')
    original_img.save(input_path)
    
    # Save restored image
    output_path = os.path.join(args.output_dir, 'restored.png')
    restored_img = save_image(restored_tensor, output_path, original_size)
    
    # Save initial restoration J_hat_1 if available
    initial_img = None
    if initial_restoration is not None:
        initial_path = os.path.join(args.output_dir, 'initial_restoration.png')
        initial_img = save_image(initial_restoration, initial_path, original_size)
    
    # Save refinement if available
    refinement_img = None
    if refinement is not None:
        refinement_path = os.path.join(args.output_dir, 'refinement.png')
        # Scale refinement for better visualization (it's usually small)
        scaled_refinement = refinement * 5.0  # Amplify by 5x for visibility
        scaled_refinement = torch.clamp(scaled_refinement + 0.5, 0, 1)  # Center around 0.5 and clamp
        refinement_img = save_image(scaled_refinement, refinement_path, original_size)
    
    # Save and visualize degradation maps with overlays
    if args.save_degradation_maps:
        # Create a numpy version of the input image for overlays
        input_np = np.array(original_img).astype(np.float32) / 255.0
        
        if degradation_maps is not None:
            visualize_degradation_maps_with_overlay(
                input_np, 
                degradation_maps.cpu().squeeze(0), 
                args.output_dir, 
                "degradation_map",
                original_size,  # Pass original size for resizing
                cmaps=['inferno', 'viridis', 'plasma', 'magma']
            )
        
        if residual_maps is not None:
            visualize_degradation_maps_with_overlay(
                input_np, 
                residual_maps.cpu().squeeze(0), 
                args.output_dir, 
                "residual_map",
                original_size,  # Pass original size for resizing
                cmaps=['cividis', 'YlOrRd', 'hot', 'coolwarm']
            )
    
    # Save individual hypotheses if requested
    if args.save_hypotheses and hypotheses:
        # Create directory for hypotheses
        hyp_dir = os.path.join(args.output_dir, "hypotheses")
        os.makedirs(hyp_dir, exist_ok=True)
        
        # Save each hypothesis
        for i, hyp in enumerate(hypotheses):
            hyp_path = os.path.join(hyp_dir, f"hypothesis_{i+1}.png")
            save_image(hyp, hyp_path, original_size)  # Resize to original image size
    
    # Calculate unsupervised metrics
    print("\nCalculating unsupervised metrics...")
    
    # Convert images to numpy for metrics calculation
    input_np = np.array(original_img).astype(np.float32) / 255.0
    restored_np = np.array(restored_img).astype(np.float32) / 255.0
    
    # For input image
    input_uiqm = calculate_uiqm(input_np)
    input_uism = calculate_uism(input_np)
    input_brisque = calculate_brisque(input_np)
    
    # For restored image
    restored_uiqm = calculate_uiqm(restored_np)
    restored_uism = calculate_uism(restored_np)
    restored_brisque = calculate_brisque(restored_np)
    
    # Print results
    print("\nInput Image Metrics:")
    print(f"UIQM: {input_uiqm:.4f}")
    print(f"UISM: {input_uism:.4f}")
    print(f"BRISQUE: {input_brisque:.4f} (lower is better)")
    
    print("\nRestored Image Metrics:")
    print(f"UIQM: {restored_uiqm:.4f}")
    print(f"UISM: {restored_uism:.4f}")
    print(f"BRISQUE: {restored_brisque:.4f} (lower is better)")
    
    # Calculate improvements
    uiqm_improvement = restored_uiqm - input_uiqm
    uism_improvement = restored_uism - input_uism
    brisque_improvement = input_brisque - restored_brisque  # For BRISQUE, lower is better
    
    print("\nImprovements:")
    print(f"UIQM: {'+' if uiqm_improvement >= 0 else ''}{uiqm_improvement:.4f}")
    print(f"UISM: {'+' if uism_improvement >= 0 else ''}{uism_improvement:.4f}")
    print(f"BRISQUE: {'+' if brisque_improvement >= 0 else ''}{brisque_improvement:.4f}")
    
    # Collect metrics for visualizations
    metrics = {
        'input_uiqm': input_uiqm,
        'input_uism': input_uism,
        'input_brisque': input_brisque,
        'restored_uiqm': restored_uiqm,
        'restored_uism': restored_uism,
        'restored_brisque': restored_brisque,
        'uiqm_improvement': uiqm_improvement,
        'uism_improvement': uism_improvement,
        'brisque_improvement': brisque_improvement
    }
    
    # Create visualizations
    if args.visualize:
        # Main comparison visualization
        visualize_results(
            original_img, 
            restored_img, 
            args.output_dir, 
            initial_img=initial_img, 
            metrics=metrics,
            refinement=refinement_img,
            hypotheses=hypotheses
        )
        
        # Progressive improvement comparison if available
        if initial_img is not None:
            compare_progressive_improvement(
                original_img,
                initial_img,
                restored_img,
                args.output_dir,
                metrics=metrics
            )
    
    # Save metrics to a text file
    with open(os.path.join(args.output_dir, 'metrics.txt'), 'w') as f:
        f.write("Underwater Image Restoration Metrics\n")
        f.write("================================\n\n")
        
        f.write(f"Input Image: {args.input_path}\n")
        f.write(f"Model Checkpoint: {args.progressive_checkpoint}\n\n")
        
        f.write("Input Image Metrics:\n")
        f.write(f"UIQM: {input_uiqm:.4f}\n")
        f.write(f"UISM: {input_uism:.4f}\n")
        f.write(f"BRISQUE: {input_brisque:.4f} (lower is better)\n\n")
        
        f.write("Restored Image Metrics:\n")
        f.write(f"UIQM: {restored_uiqm:.4f}\n")
        f.write(f"UISM: {restored_uism:.4f}\n")
        f.write(f"BRISQUE: {restored_brisque:.4f} (lower is better)\n\n")
        
        f.write("Improvements:\n")
        f.write(f"UIQM: {'+' if uiqm_improvement >= 0 else ''}{uiqm_improvement:.4f}\n")
        f.write(f"UISM: {'+' if uism_improvement >= 0 else ''}{uism_improvement:.4f}\n")
        f.write(f"BRISQUE: {'+' if brisque_improvement >= 0 else ''}{brisque_improvement:.4f}\n")
    
    print(f"\nEvaluation completed. Results saved to {args.output_dir}")

if __name__ == '__main__':
    main()
    