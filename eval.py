import os
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision.utils as vutils

from utils import AverageMeter, calculate_metrics, create_directory, visualize_results, compare_progressive_improvement
from losses import CombinedLoss, RefinementLoss


def evaluate(model, val_loader, device, args, is_progressive=False):
    """Evaluate the model on the validation set"""
    model.eval()
    
    # Metrics
    metrics = {
        'psnr': AverageMeter(),
        'ssim': AverageMeter()
    }
    
    # Progressive model specific metrics
    if is_progressive:
        progressive_metrics = {
            'psnr_initial': AverageMeter(),
            'ssim_initial': AverageMeter(),
            'psnr_improvement': AverageMeter(),
            'ssim_improvement': AverageMeter()
        }
    else:
        progressive_metrics = None
    
    # Create output directories if saving results
    if args.save_images:
        output_img_dir = os.path.join(args.output_dir, 'restored_images')
        create_directory(output_img_dir)
    
    if args.save_hypotheses:
        output_hyp_dir = os.path.join(args.output_dir, 'hypotheses')
        create_directory(output_hyp_dir)
    
    if args.save_degradation_maps:
        output_deg_dir = os.path.join(args.output_dir, 'degradation_maps')
        create_directory(output_deg_dir)
    
    if args.save_refinement:
        output_ref_dir = os.path.join(args.output_dir, 'refinement')
        create_directory(output_ref_dir)
        output_res_dir = os.path.join(args.output_dir, 'residual_maps')
        create_directory(output_res_dir)
    
    if args.visualize:
        output_vis_dir = os.path.join(args.output_dir, 'visualizations')
        create_directory(output_vis_dir)
    
    # Evaluation loop
    pbar = tqdm(val_loader)
    for i, batch in enumerate(pbar):
        # Get data
        degraded = batch['degraded'].to(device)
        reference = batch['reference'].to(device)
        filenames = batch['filename']
        
        # Forward pass
        with torch.no_grad():
            outputs = model(degraded)
        
        # Calculate metrics
        batch_metrics = calculate_metrics(outputs['restored_image'], reference)
        for k, v in batch_metrics.items():
            metrics[k].update(v)
        
        # Calculate progressive model metrics if applicable
        if progressive_metrics is not None and 'initial_restoration' in outputs:
            initial_metrics = calculate_metrics(outputs['initial_restoration'], reference)
            progressive_metrics['psnr_initial'].update(initial_metrics['psnr'])
            progressive_metrics['ssim_initial'].update(initial_metrics['ssim'])
            
            # Calculate improvements
            psnr_improvement = batch_metrics['psnr'] - initial_metrics['psnr']
            ssim_improvement = batch_metrics['ssim'] - initial_metrics['ssim']
            
            progressive_metrics['psnr_improvement'].update(psnr_improvement)
            progressive_metrics['ssim_improvement'].update(ssim_improvement)
        
        # Update progress bar
        desc = f"Evaluation | PSNR: {metrics['psnr'].avg:.2f} | SSIM: {metrics['ssim'].avg:.4f}"
        if progressive_metrics is not None:
            desc += f" | Improvement: {progressive_metrics['psnr_improvement'].avg:.2f}dB"
        pbar.set_description(desc)
        
        # Save results if requested
        for j, filename in enumerate(filenames):
            # Create base filename without extension
            base_filename = os.path.splitext(filename)[0]
            
            # Save restored image
            if args.save_images:
                vutils.save_image(
                    outputs['restored_image'][j],
                    os.path.join(output_img_dir, f"{base_filename}_restored.png")
                )
            
            # Save individual hypotheses
            if args.save_hypotheses:
                for h_idx, hypothesis in enumerate(outputs['hypotheses']):
                    vutils.save_image(
                        hypothesis[j],
                        os.path.join(output_hyp_dir, f"{base_filename}_hypothesis_{h_idx}.png")
                    )
            
            # Save degradation maps
            if args.save_degradation_maps:
                for d_idx in range(outputs['degradation_maps'].size(1)):
                    # Convert single-channel map to RGB for visualization
                    deg_map = outputs['degradation_maps'][j, d_idx].unsqueeze(0).repeat(3, 1, 1)
                    vutils.save_image(
                        deg_map,
                        os.path.join(output_deg_dir, f"{base_filename}_degradation_{d_idx}.png"),
                        normalize=True
                    )
            
            # Save refinement results if applicable and requested
            if args.save_refinement and 'initial_restoration' in outputs and 'refinement' in outputs:
                # Save initial restoration J_hat_1
                vutils.save_image(
                    outputs['initial_restoration'][j],
                    os.path.join(output_ref_dir, f"{base_filename}_initial.png")
                )
                
                # Save refinement map (amplified for visibility)
                refinement = outputs['refinement'][j]
                # Scale refinement for visibility
                visible_refinement = refinement * 5 + 0.5
                visible_refinement = torch.clamp(visible_refinement, 0, 1)
                vutils.save_image(
                    visible_refinement,
                    os.path.join(output_ref_dir, f"{base_filename}_refinement.png")
                )
                
                # Save residual degradation maps Mr maps
                if 'residual_maps' in outputs:
                    for r_idx in range(outputs['residual_maps'].size(1)):
                        # Convert single-channel map to RGB for visualization
                        res_map = outputs['residual_maps'][j, r_idx].unsqueeze(0).repeat(3, 1, 1)
                        vutils.save_image(
                            res_map,
                            os.path.join(output_res_dir, f"{base_filename}_residual_{r_idx}.png"),
                            normalize=True
                        )
            
            # Create visualizations if requested
            if args.visualize and j < 10:  # Limit to first 10 images
                # Convert tensors to numpy for visualization
                degraded_np = degraded[j].detach().cpu().numpy().transpose(1, 2, 0)
                restored_np = outputs['restored_image'][j].detach().cpu().numpy().transpose(1, 2, 0)
                reference_np = reference[j].detach().cpu().numpy().transpose(1, 2, 0)
                
                # Convert degradation maps to numpy
                if 'degradation_maps' in outputs:
                    degradation_maps_np = outputs['degradation_maps'][j].detach().cpu().numpy()
                else:
                    degradation_maps_np = None
                
                # Convert hypotheses to numpy
                if 'hypotheses' in outputs:
                    hypotheses_np = [hyp[j].detach().cpu().numpy().transpose(1, 2, 0) 
                                    for hyp in outputs['hypotheses']]
                else:
                    hypotheses_np = None
                
                # For progressive model
                if 'initial_restoration' in outputs and 'refinement' in outputs:
                    initial_np = outputs['initial_restoration'][j].detach().cpu().numpy().transpose(1, 2, 0)
                    refinement_np = outputs['refinement'][j].detach().cpu().numpy().transpose(1, 2, 0)
                    
                    # Convert residual maps to numpy
                    if 'residual_maps' in outputs:
                        residual_maps_np = outputs['residual_maps'][j].detach().cpu().numpy()
                    else:
                        residual_maps_np = None
                    
                    # Create visualization
                    visualize_results(
                        degraded=degraded_np,
                        restored=restored_np,
                        reference=reference_np,
                        degradation_maps=degradation_maps_np,
                        hypotheses=hypotheses_np,
                        initial_restoration=initial_np,
                        refinement=refinement_np,
                        residual_maps=residual_maps_np,
                        save_path=os.path.join(output_vis_dir, f"{base_filename}_progressive_vis.png")
                    )
                    
                    # Create progressive improvement comparison
                    current_metrics = {}
                    if progressive_metrics is not None:
                        current_metrics = {
                            'psnr_initial': calculate_metrics(
                                outputs['initial_restoration'][j].unsqueeze(0), 
                                reference[j].unsqueeze(0)
                            )['psnr'],
                            'ssim_initial': calculate_metrics(
                                outputs['initial_restoration'][j].unsqueeze(0), 
                                reference[j].unsqueeze(0)
                            )['ssim'],
                            'psnr': calculate_metrics(
                                outputs['restored_image'][j].unsqueeze(0), 
                                reference[j].unsqueeze(0)
                            )['psnr'],
                            'ssim': calculate_metrics(
                                outputs['restored_image'][j].unsqueeze(0), 
                                reference[j].unsqueeze(0)
                            )['ssim'],
                            'psnr_improvement': calculate_metrics(
                                outputs['restored_image'][j].unsqueeze(0), 
                                reference[j].unsqueeze(0)
                            )['psnr'] - calculate_metrics(
                                outputs['initial_restoration'][j].unsqueeze(0), 
                                reference[j].unsqueeze(0)
                            )['psnr'],
                            'ssim_improvement': calculate_metrics(
                                outputs['restored_image'][j].unsqueeze(0), 
                                reference[j].unsqueeze(0)
                            )['ssim'] - calculate_metrics(
                                outputs['initial_restoration'][j].unsqueeze(0), 
                                reference[j].unsqueeze(0)
                            )['ssim']
                        }
                    
                    compare_progressive_improvement(
                        degraded=degraded_np,
                        initial_restoration=initial_np,
                        final_restoration=restored_np,
                        reference=reference_np,
                        save_path=os.path.join(output_vis_dir, f"{base_filename}_improvement.png"),
                        metrics=current_metrics
                    )
                else:
                    # Standard visualization
                    visualize_results(
                        degraded=degraded_np,
                        restored=restored_np,
                        reference=reference_np,
                        degradation_maps=degradation_maps_np,
                        hypotheses=hypotheses_np,
                        save_path=os.path.join(output_vis_dir, f"{base_filename}_vis.png")
                    )
    
    # Calculate final metrics
    final_metrics = {
        'psnr': metrics['psnr'].avg,
        'ssim': metrics['ssim'].avg
    }
    
    # Add progressive metrics if applicable
    if progressive_metrics is not None:
        for k, v in progressive_metrics.items():
            final_metrics[k] = v.avg
    
    # Save metrics to file
    with open(os.path.join(args.output_dir, 'metrics.txt'), 'w') as f:
        f.write("Evaluation Results\n")
        f.write("=================\n\n")
        for k, v in final_metrics.items():
            f.write(f"{k}: {v:.4f}\n")
    
    return final_metrics
