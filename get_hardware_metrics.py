import time
import argparse
import torch
import torch.nn as nn
from thop import profile, clever_format
from ptflops import get_model_complexity_info

from model import DGMHRN, TIDE
from utils import load_checkpoint

def parse_args():
    parser = argparse.ArgumentParser(description='Hardware performance analysis')
    
    # Model arguments
    parser.add_argument('--base_checkpoint', type=str, required=True, help='Path to base model checkpoint')
    parser.add_argument('--progressive_checkpoint', type=str, required=True, help='Path to progressive model checkpoint')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    
    # Model parameters
    parser.add_argument('--base_channels', type=int, default=64, help='Base number of channels')
    parser.add_argument('--num_downs', type=int, default=5, help='Number of downsampling layers')
    parser.add_argument('--num_degradation_types', type=int, default=4, help='Number of degradation types (color, contrast, detail, noise)')
    parser.add_argument('--norm_type', type=str, default='instance', help='Normalization type')
    parser.add_argument('--activation', type=str, default='leaky_relu', help='Activation function')
    parser.add_argument('--fusion_type', type=str, default='learned', help='Fusion mechanism type')
    
    # Benchmark settings
    parser.add_argument('--img_size', type=int, default=512, help='Image size for testing')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for testing')
    parser.add_argument('--num_runs', type=int, default=100, help='Number of inference runs for averaging')
    parser.add_argument('--warm_up', type=int, default=10, help='Number of warm-up runs before measurement')
    
    args = parser.parse_args()
    return args

def load_models(args, device):
    """Load both base and progressive models"""
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
    
    # Create a separate base model for the progressive model
    base_model_for_progressive = DGMHRN(
        in_channels=3,
        base_channels=args.base_channels,
        num_downs=args.num_downs,
        num_degradation_types=args.num_degradation_types,
        norm_type=args.norm_type,
        activation=args.activation,
        fusion_type=args.fusion_type
    ).to(device)
    
    # Create progressive model
    progressive_model = TIDE(
        base_model=base_model_for_progressive,
        num_degradation_types=args.num_degradation_types,
        norm_type=args.norm_type,
        activation=args.activation
    ).to(device)
    
    # Load base model checkpoint
    load_checkpoint(args.base_checkpoint, base_model)
    print(f"Loaded base model from: {args.base_checkpoint}")
    
    # Load progressive model checkpoint
    load_checkpoint(args.progressive_checkpoint, progressive_model)
    print(f"Loaded progressive model from: {args.progressive_checkpoint}")
    
    # Set both models to evaluation mode
    base_model.eval()
    progressive_model.eval()
    
    return base_model, progressive_model

def count_parameters(model):
    """Count the number of trainable parameters in a model"""
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params

def measure_inference_time(model, input_tensor, num_runs=100, warm_up=10):
    """Measure average inference time over multiple runs"""
    device = next(model.parameters()).device
    
    # Warm-up runs
    with torch.no_grad():
        for _ in range(warm_up):
            _ = model(input_tensor)
            if device.type == 'cuda':
                torch.cuda.synchronize()
    
    # Timed runs
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(input_tensor)
            if device.type == 'cuda':
                torch.cuda.synchronize()
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_runs
    return avg_time

def calculate_metrics(model, input_tensor, img_size, num_runs, warm_up):
    """Calculate performance metrics for a model"""
    # Count parameters
    num_params = count_parameters(model)
    
    # Calculate FLOPs
    macs, _ = profile(model, inputs=(input_tensor,))
    macs_str, _ = clever_format([macs, num_params], "%.3f")
    
    flops_count, _ = get_model_complexity_info(
        model, (3, img_size, img_size), 
        as_strings=False, print_per_layer_stat=False, verbose=False
    )
    gflops = flops_count * 2 / 1e9  # Convert MACs to GFLOPs
    
    # Measure inference time
    inference_time = measure_inference_time(model, input_tensor, num_runs, warm_up)
    inference_time_ms = inference_time * 1000
    
    # Calculate FPS
    fps = input_tensor.size(0) / inference_time
    
    return {
        'parameters': num_params,
        'macs': macs_str,
        'gflops': gflops,
        'inference_time_ms': inference_time_ms,
        'fps': fps
    }

def main():
    # Parse arguments
    args = parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Using device: {device}")
    
    # Load models
    base_model, progressive_model = load_models(args, device)
    
    # Create a random input tensor
    input_tensor = torch.randn(args.batch_size, 3, args.img_size, args.img_size).to(device)
    
    # Calculate metrics for base model
    print("\nEvaluating Base Model...")
    base_metrics = calculate_metrics(base_model, input_tensor, args.img_size, args.num_runs, args.warm_up)
    
    # Calculate metrics for progressive model
    print("\nEvaluating Progressive Model...")
    prog_metrics = calculate_metrics(progressive_model, input_tensor, args.img_size, args.num_runs, args.warm_up)
    
    # Print results in a table format
    print("\n" + "="*70)
    print(f"Performance Comparison (Image Size: {args.img_size}x{args.img_size}, Batch Size: {args.batch_size})")
    print("="*70)
    
    metrics = [
        ("Parameters", f"{base_metrics['parameters']:,}", f"{prog_metrics['parameters']:,}"),
        ("MACs", base_metrics['macs'], prog_metrics['macs']),
        ("GFLOPs", f"{base_metrics['gflops']:.4f}", f"{prog_metrics['gflops']:.4f}"),
        ("Inference Time (ms)", f"{base_metrics['inference_time_ms']:.2f}", f"{prog_metrics['inference_time_ms']:.2f}"),
        ("Throughput (FPS)", f"{base_metrics['fps']:.2f}", f"{prog_metrics['fps']:.2f}")
    ]
    
    # Calculate percent differences
    percent_diffs = []
    for key in ['parameters', 'gflops', 'inference_time_ms', 'fps']:
        if base_metrics[key] != 0:
            diff = (prog_metrics[key] - base_metrics[key]) / base_metrics[key] * 100
            percent_diffs.append(f"{diff:+.2f}%")
        else:
            percent_diffs.append("N/A")
    
    # Print the table
    col_width = [25, 20, 20]
    header = ["Metric", "Base Model", "Progressive Model"]
    print(" | ".join(h.ljust(col_width[i]) for i, h in enumerate(header)))
    print("-" * 70)
    
    for i, (metric, base_val, prog_val) in enumerate(metrics):
        row = [metric, base_val, prog_val]
        print(" | ".join(str(r).ljust(col_width[i]) for i, r in enumerate(row)))
    
    # Print percent differences
    print("\nPercent Differences (Progressive vs Base):")
    diff_metrics = ["Parameters", "GFLOPs", "Inference Time", "Throughput (FPS)"]
    for metric, diff in zip(diff_metrics, percent_diffs):
        print(f"{metric}: {diff}")

if __name__ == '__main__':
    main()
    