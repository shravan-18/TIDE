import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
from tqdm import tqdm
import numpy as np

from model import DGMHRN, TIDE
from dataset import get_dataloaders
from losses import CombinedLoss, RefinementLoss, ProgressiveCombinedLoss
from utils import AverageMeter, calculate_metrics, save_checkpoint, load_checkpoint, save_epoch_images


def train_epoch(model, train_loader, criterion, optimizer, epoch, device, 
                scaler=None, args=None, writer=None, log_dir=None):
    """Train for one epoch"""
    model.train()
    
    # Metrics
    batch_time = AverageMeter()
    data_time = AverageMeter()
    
    # Dynamically initialize losses dictionary based on first iteration output
    losses = {'total': AverageMeter()}
    
    # For progressive model, we'll track different metrics
    is_progressive = hasattr(model, 'enable_refinement')
    
    # Standard metrics for all models
    metrics = {
        'psnr': AverageMeter(),
        'ssim': AverageMeter()
    }
    
    start = time.time()
    pbar = tqdm(train_loader)
    
    for i, batch in enumerate(pbar):
        data_time.update(time.time() - start)
        
        # Get data
        degraded = batch['degraded'].to(device)
        reference = batch['reference'].to(device)
        
        # Forward pass
        if args.mixed_precision:
            with torch.cuda.amp.autocast():
                outputs = model(degraded)
                loss_dict = criterion(outputs, reference, degraded)
        else:
            outputs = model(degraded)
            loss_dict = criterion(outputs, reference, degraded)
        
        # Initialize loss trackers if this is the first batch
        if i == 0:
            # Create a meter for each loss component
            for k in loss_dict.keys():
                if k not in losses:
                    losses[k] = AverageMeter()
        
        # Calculate metrics
        with torch.no_grad():
            batch_metrics = calculate_metrics(outputs['restored_image'], reference)
            for k, v in batch_metrics.items():
                metrics[k].update(v)
        
        # Backward pass
        optimizer.zero_grad()
        
        if args.mixed_precision:
            scaler.scale(loss_dict['total']).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_dict['total'].backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
        
        # Update losses
        for k, v in loss_dict.items():
            if isinstance(v, torch.Tensor) and k in losses:
                losses[k].update(v.item())
        
        # Measure elapsed time
        batch_time.update(time.time() - start)
        start = time.time()
        
        # Log progress
        if i % args.log_interval == 0:
            pbar.set_description(
                f"Epoch {epoch} | Loss: {losses['total'].avg:.4f} | "
                f"PSNR: {metrics['psnr'].avg:.2f} | SSIM: {metrics['ssim'].avg:.4f}"
            )
            
            # Log to tensorboard
            if writer is not None:
                global_step = epoch * len(train_loader) + i
                writer.add_scalar('train/loss', losses['total'].avg, global_step)
                for k, v in losses.items():
                    if k != 'total':
                        writer.add_scalar(f'train/loss_{k}', v.avg, global_step)
                for k, v in metrics.items():
                    writer.add_scalar(f'train/{k}', v.avg, global_step)
                
                # Log learning rate
                writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], global_step)
                
                # Log images
                if i % (args.log_interval * 10) == 0:
                    # Take the first few samples from the batch
                    n_vis = min(args.vis_samples, degraded.size(0))
                    
                    degraded_vis = degraded[:n_vis]
                    reference_vis = reference[:n_vis]
                    restored_vis = outputs['restored_image'][:n_vis]
                    
                    # Check if this is a progressive model
                    if 'initial_restoration' in outputs and 'refinement' in outputs:
                        # Add initial restoration J_hat_1 and refinement to visualization
                        initial_vis = outputs['initial_restoration'][:n_vis]
                        refinement_vis = outputs['refinement'][:n_vis]
                        
                        # Scale refinement for visibility (it's usually small)
                        refinement_vis = refinement_vis * 5 + 0.5
                        refinement_vis = torch.clamp(refinement_vis, 0, 1)
                        
                        # Visualization grid: degraded, initial, refinement, restored, reference
                        vis_images = [degraded_vis, initial_vis, refinement_vis, restored_vis, reference_vis]
                        
                        # Also add residual degradation maps Mr maps
                        if 'residual_maps' in outputs:
                            residual_maps_vis = outputs['residual_maps'][:n_vis]
                            res_grid = vutils.make_grid(residual_maps_vis, nrow=n_vis, normalize=True)
                            # writer.add_image('train/residual_maps', res_grid, global_step)
                    else:
                        # Get individual hypotheses
                        hypotheses_vis = []
                        for hyp in outputs['hypotheses']:
                            hypotheses_vis.append(hyp[:n_vis])
                        
                        # Standard visualization: degraded, hypotheses, restored, reference
                        vis_images = [degraded_vis] + hypotheses_vis + [restored_vis, reference_vis]
                    
                    vis_grid = torch.cat(vis_images, dim=0)
                    vis_grid = vutils.make_grid(vis_grid, nrow=n_vis, normalize=True)
                    # writer.add_image('train/results', vis_grid, global_step)
                    
                    # Log degradation maps
                    degradation_maps = outputs['degradation_maps'][:n_vis]
                    deg_grid = vutils.make_grid(degradation_maps, nrow=n_vis, normalize=True)
                    # writer.add_image('train/degradation_maps', deg_grid, global_step)
    
    # Save images after the epoch if enabled
    if args.save_epoch_images:
        save_epoch_images(
            degraded=degraded,
            restored=outputs['restored_image'],
            reference=reference,
            hypotheses=outputs.get('hypotheses', None),
            degradation_maps=outputs.get('degradation_maps', None),
            initial_restoration=outputs.get('initial_restoration', None),
            refinement=outputs.get('refinement', None),
            residual_maps=outputs.get('residual_maps', None),
            save_dir=args.save_dir,
            epoch=epoch,
            phase='train'
        )
    
    # Return average metrics
    return {
        'loss': losses['total'].avg,
        'psnr': metrics['psnr'].avg,
        'ssim': metrics['ssim'].avg
    }


def validate(model, val_loader, criterion, epoch, device, args=None, writer=None, log_dir=None):
    """Validate the model"""
    model.eval()
    
    # Metrics
    losses = {
        'total': AverageMeter(),
        'l1': AverageMeter(),
        'ssim': AverageMeter(),
        'perceptual': AverageMeter(),
        'diversity': AverageMeter(),
        'degradation': AverageMeter(),
        'hypotheses': AverageMeter()
    }
    metrics = {
        'psnr': AverageMeter(),
        'ssim': AverageMeter()
    }
    
    # Progressive model specific metrics
    if hasattr(model, 'enable_refinement'):
        progressive_metrics = {
            'psnr_initial': AverageMeter(),
            'ssim_initial': AverageMeter(),
            'psnr_improvement': AverageMeter(),
            'ssim_improvement': AverageMeter()
        }
    else:
        progressive_metrics = None
    
    pbar = tqdm(val_loader)
    
    with torch.no_grad():
        for i, batch in enumerate(pbar):
            # Get data
            degraded = batch['degraded'].to(device)
            reference = batch['reference'].to(device)
            filenames = batch['filename']
            
            # Forward pass
            outputs = model(degraded)
            loss_dict = criterion(outputs, reference, degraded)
            
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
            
            # Update losses
            for k, v in loss_dict.items():
                if isinstance(v, torch.Tensor) and k in losses:
                    losses[k].update(v.item())
            
            # Log progress
            desc = f"Validation | Loss: {losses['total'].avg:.4f} | PSNR: {metrics['psnr'].avg:.2f} | SSIM: {metrics['ssim'].avg:.4f}"
            if progressive_metrics is not None:
                desc += f" | Improv: {progressive_metrics['psnr_improvement'].avg:.2f}dB"
            pbar.set_description(desc)
            
            # Save some validation results for visualization
            if i == 0 and writer is not None:
                # Take the first few samples from the batch
                n_vis = min(args.vis_samples, degraded.size(0))
                
                degraded_vis = degraded[:n_vis]
                reference_vis = reference[:n_vis]
                restored_vis = outputs['restored_image'][:n_vis]
                
                # Check if this is a progressive model
                if 'initial_restoration' in outputs and 'refinement' in outputs:
                    # Add initial restoration J_hat_1 and refinement to visualization
                    initial_vis = outputs['initial_restoration'][:n_vis]
                    refinement_vis = outputs['refinement'][:n_vis]
                    
                    # Scale refinement for visibility (it's usually small)
                    refinement_vis = refinement_vis * 5 + 0.5
                    refinement_vis = torch.clamp(refinement_vis, 0, 1)
                    
                    # Visualization grid: degraded, initial, refinement, restored, reference
                    vis_images = [degraded_vis, initial_vis, refinement_vis, restored_vis, reference_vis]
                    
                    # Also add residual degradation maps Mr maps
                    if 'residual_maps' in outputs:
                        residual_maps_vis = outputs['residual_maps'][:n_vis]
                        res_grid = vutils.make_grid(residual_maps_vis, nrow=n_vis, normalize=True)
                        # writer.add_image('val/residual_maps', res_grid, epoch)
                else:
                    # Get individual hypotheses
                    hypotheses_vis = []
                    for hyp in outputs['hypotheses']:
                        hypotheses_vis.append(hyp[:n_vis])
                    
                    # Standard visualization: degraded, hypotheses, restored, reference
                    vis_images = [degraded_vis] + hypotheses_vis + [restored_vis, reference_vis]
                
                vis_grid = torch.cat(vis_images, dim=0)
                vis_grid = vutils.make_grid(vis_grid, nrow=n_vis, normalize=True)
                # writer.add_image('val/results', vis_grid, epoch)
                
                # Log degradation maps
                degradation_maps = outputs['degradation_maps'][:n_vis]
                deg_grid = vutils.make_grid(degradation_maps, nrow=n_vis, normalize=True)
                # writer.add_image('val/degradation_maps', deg_grid, epoch)
    
    # Save images after validation if enabled
    if args.save_epoch_images:
        save_epoch_images(
            degraded=degraded,
            restored=outputs['restored_image'],
            reference=reference,
            hypotheses=outputs['hypotheses'],
            degradation_maps=outputs['degradation_maps'],
            initial_restoration=outputs.get('initial_restoration', None),
            refinement=outputs.get('refinement', None),
            residual_maps=outputs.get('residual_maps', None),
            save_dir=args.save_dir,
            epoch=epoch,
            phase='val'
        )
    
    # Log validation metrics
    if writer is not None:
        writer.add_scalar('val/loss', losses['total'].avg, epoch)
        for k, v in losses.items():
            if k != 'total':
                writer.add_scalar(f'val/loss_{k}', v.avg, epoch)
        for k, v in metrics.items():
            writer.add_scalar(f'val/{k}', v.avg, epoch)
        
        # Log progressive metrics if applicable
        if progressive_metrics is not None:
            for k, v in progressive_metrics.items():
                writer.add_scalar(f'val/{k}', v.avg, epoch)
    
    # Print validation results
    print(f"Validation Epoch {epoch} | Loss: {losses['total'].avg:.4f} | "
          f"PSNR: {metrics['psnr'].avg:.2f} | SSIM: {metrics['ssim'].avg:.4f}")
    
    if progressive_metrics is not None:
        print(f"Progressive metrics | Initial PSNR: {progressive_metrics['psnr_initial'].avg:.2f} | "
              f"PSNR Improvement: {progressive_metrics['psnr_improvement'].avg:.2f}dB | "
              f"SSIM Improvement: {progressive_metrics['ssim_improvement'].avg:.4f}")
    
    # Return average metrics
    result = {
        'loss': losses['total'].avg,
        'psnr': metrics['psnr'].avg,
        'ssim': metrics['ssim'].avg
    }
    
    # Add progressive metrics if applicable
    if progressive_metrics is not None:
        for k, v in progressive_metrics.items():
            result[k] = v.avg
    
    return result


def train_model(model, train_loader, val_loader, device, args):
    """Train the model with the given arguments"""
    # Set up tensorboard
    writer = SummaryWriter(log_dir=args.log_dir)
    
    # Create loss function
    criterion = CombinedLoss(
        lambda_l1=args.lambda_l1,
        lambda_ssim=args.lambda_ssim,
        lambda_perceptual=args.lambda_perceptual,
        lambda_diversity=args.lambda_diversity,
        lambda_degradation=args.lambda_degradation
    ).to(device)
    
    # Create optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Create learning rate scheduler
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=args.lr_cycle_epochs * len(train_loader),
        T_mult=args.lr_cycle_mult,
        eta_min=args.lr * 0.01
    )
    
    # Set up mixed precision training
    scaler = torch.cuda.amp.GradScaler() if args.mixed_precision else None
    
    # Resume from checkpoint if specified
    start_epoch = args.start_epoch
    best_psnr = 0
    if args.resume:
        checkpoint = load_checkpoint(args.resume, model, optimizer, scheduler)
        start_epoch = checkpoint['epoch'] + 1
        best_psnr = checkpoint.get('best_psnr', 0)
        print(f"Resuming from epoch {start_epoch} with best PSNR {best_psnr:.2f}")
    
    # Train loop
    for epoch in range(start_epoch, args.num_epochs):
        print(f"Epoch {epoch}/{args.num_epochs}")
        
        # Train for one epoch
        train_metrics = train_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            epoch=epoch,
            device=device,
            scaler=scaler,
            args=args,
            writer=writer,
            log_dir=args.log_dir
        )
        
        # Update learning rate
        scheduler.step()
        
        # Validate
        if (epoch + 1) % args.val_interval == 0:
            val_metrics = validate(
                model=model,
                val_loader=val_loader,
                criterion=criterion,
                epoch=epoch,
                device=device,
                args=args,
                writer=writer,
                log_dir=args.log_dir
            )
            
            # Save best model
            if val_metrics['psnr'] > best_psnr:
                best_psnr = val_metrics['psnr']
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    metrics=val_metrics,
                    best_psnr=best_psnr,
                    path=os.path.join(args.save_dir, 'best_model.pth')
                )
                print(f"Saved best model with PSNR {best_psnr:.2f}")
        
        # Save checkpoint periodically
        if (epoch + 1) % args.save_interval == 0:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                metrics=train_metrics,
                best_psnr=best_psnr,
                path=os.path.join(args.save_dir, f'checkpoint_epoch_{epoch}.pth')
            )
            print(f"Saved checkpoint for epoch {epoch}")
    
    # Save final model
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=args.num_epochs - 1,
        metrics=train_metrics,
        best_psnr=best_psnr,
        path=os.path.join(args.save_dir, 'final_model.pth')
    )
    print(f"Training completed. Final model saved.")
    
    # Close tensorboard writer
    writer.close()


def train_refinement(base_model, train_loader, val_loader, device, args):
    """Train the progressive refinement stage with the base model frozen"""
    print("Creating two-stage restoration network with frozen base model...")
    
    # Create the progressive model
    progressive_model = TIDE(
        base_model=base_model,
        num_degradation_types=args.num_degradation_types,
        norm_type=args.norm_type,
        activation=args.activation
    ).to(device)

    try:
        checkpoint = torch.load(r'checkpoints\\refinement\best_refinement_model.pth', weights_only=False)

        # Create dummy input for forward pass to initialize dynamic layers
        dummy_input = torch.zeros(1, 3, 256, 256)
        if torch.cuda.is_available() and next(progressive_model.parameters()).is_cuda:
            dummy_input = dummy_input.cuda()
        
        # Run a forward pass with dummy data to initialize dynamic layers
        original_mode = progressive_model.training
        progressive_model.eval()
        with torch.no_grad():
            _ = progressive_model(dummy_input)
        if original_mode:
            progressive_model.train()

        progressive_model.load_state_dict(checkpoint['model_state_dict'])
        print("Restored refinement model state successfully")
    except Exception as e:
        print("There was an error loading the model: ", e)
    
    # Freeze the base model
    for param in progressive_model.base_model.parameters():
        param.requires_grad = False
    
    # Create refinement-specific directories
    writer = SummaryWriter(log_dir=os.path.join(args.log_dir, 'refinement'))
    refinement_save_dir = os.path.join(args.save_dir, 'refinement')
    os.makedirs(refinement_save_dir, exist_ok=True)
    
    # Use the refinement loss
    criterion = RefinementLoss(
        lambda_recon=args.lambda_recon,
        lambda_magnitude=args.lambda_magnitude,
        lambda_improve=args.lambda_improve,
        lambda_ssim=args.lambda_ssim,
        lambda_perceptual=args.lambda_perceptual
    ).to(device)
    
    # Only optimize the refinement parameters
    refinement_params = [p for name, p in progressive_model.named_parameters() 
                         if p.requires_grad]
    
    print(f"Training {len(refinement_params)} refinement parameters, "
          f"base model parameters frozen.")
    
    # Create optimizer
    optimizer = optim.Adam(
        refinement_params,
        lr=args.refinement_lr,
        weight_decay=args.weight_decay
    )
    
    # Create learning rate scheduler
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=args.refinement_epochs // 2,
        T_mult=1,
        eta_min=args.refinement_lr * 0.01
    )
    
    # Set up mixed precision training
    scaler = torch.cuda.amp.GradScaler() if args.mixed_precision else None
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_psnr = 0
    if args.refinement_resume:
        checkpoint = load_checkpoint(args.refinement_resume, progressive_model, optimizer, scheduler)
        start_epoch = checkpoint['epoch'] + 1
        best_psnr = checkpoint.get('best_psnr', 0)
        print(f"Resuming refinement from epoch {start_epoch} with best PSNR {best_psnr:.2f}")
    
    # Modified args for refinement training
    refinement_args = args
    refinement_args.log_dir = os.path.join(args.log_dir, 'refinement')
    refinement_args.save_dir = refinement_save_dir
    refinement_args.num_epochs = args.refinement_epochs
    
    # Train loop
    for epoch in range(start_epoch, args.refinement_epochs):
        print(f"Refinement Epoch {epoch}/{args.refinement_epochs}")
        
        # Train for one epoch
        train_metrics = train_epoch(
            model=progressive_model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            epoch=epoch,
            device=device,
            scaler=scaler,
            args=refinement_args,
            writer=writer,
            log_dir=refinement_args.log_dir
        )
        
        # Update learning rate
        scheduler.step()
        
        # Validate
        if (epoch + 1) % args.val_interval == 0:
            val_metrics = validate(
                model=progressive_model,
                val_loader=val_loader,
                criterion=criterion,
                epoch=epoch,
                device=device,
                args=refinement_args,
                writer=writer,
                log_dir=refinement_args.log_dir
            )
            
            # Save best model
            if val_metrics['psnr'] > best_psnr:
                best_psnr = val_metrics['psnr']
                save_checkpoint(
                    model=progressive_model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    metrics=val_metrics,
                    best_psnr=best_psnr,
                    path=os.path.join(refinement_save_dir, 'best_refinement_model.pth')
                )
                print(f"Saved best refinement model with PSNR {best_psnr:.2f}")
        
        # Save checkpoint periodically
        if (epoch + 1) % args.save_interval == 0:
            save_checkpoint(
                model=progressive_model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                metrics=train_metrics,
                best_psnr=best_psnr,
                path=os.path.join(refinement_save_dir, f'refinement_checkpoint_epoch_{epoch}.pth')
            )
            print(f"Saved refinement checkpoint for epoch {epoch}")
    
    # Save final model
    save_checkpoint(
        model=progressive_model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=args.refinement_epochs - 1,
        metrics=train_metrics,
        best_psnr=best_psnr,
        path=os.path.join(refinement_save_dir, 'final_refinement_model.pth')
    )
    print(f"Refinement training completed. Final model saved.")
    
    # Close tensorboard writer
    writer.close()
    
    return progressive_model


def finetune_progressive_model(progressive_model, train_loader, val_loader, device, args):
    """Fine-tune the entire progressive model end-to-end"""
    print("Fine-tuning the entire progressive model end-to-end...")
    
    # Unfreeze the base model
    for param in progressive_model.parameters():
        param.requires_grad = True
    
    # Create fine-tuning specific directories
    writer = SummaryWriter(log_dir=os.path.join(args.log_dir, 'finetune'))
    finetune_save_dir = os.path.join(args.save_dir, 'finetune')
    os.makedirs(finetune_save_dir, exist_ok=True)
    
    # Use the combined progressive loss
    criterion = ProgressiveCombinedLoss(
        lambda_base=args.lambda_base,
        lambda_refinement=args.lambda_refinement,
        base_loss_weights={
            'lambda_l1': args.lambda_l1,
            'lambda_ssim': args.lambda_ssim,
            'lambda_perceptual': args.lambda_perceptual,
            'lambda_diversity': args.lambda_diversity,
            'lambda_degradation': args.lambda_degradation
        },
        refinement_loss_weights={
            'lambda_recon': args.lambda_recon,
            'lambda_magnitude': args.lambda_magnitude,
            'lambda_improve': args.lambda_improve,
            'lambda_ssim': args.lambda_ssim,
            'lambda_perceptual': args.lambda_perceptual
        }
    ).to(device)
    
    # Create optimizer with lower learning rate for fine-tuning
    optimizer = optim.Adam(
        progressive_model.parameters(),
        lr=args.finetune_lr,
        weight_decay=args.weight_decay
    )
    
    # Create learning rate scheduler
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=args.finetune_epochs // 2,
        T_mult=1,
        eta_min=args.finetune_lr * 0.01
    )
    
    # Set up mixed precision training
    scaler = torch.cuda.amp.GradScaler() if args.mixed_precision else None
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_psnr = 0
    if args.finetune_resume:
        checkpoint = load_checkpoint(args.finetune_resume, progressive_model, optimizer, scheduler)
        start_epoch = checkpoint['epoch'] + 1
        best_psnr = checkpoint.get('best_psnr', 0)
        print(f"Resuming fine-tuning from epoch {start_epoch} with best PSNR {best_psnr:.2f}")
    
    # Modified args for fine-tuning
    finetune_args = args
    finetune_args.log_dir = os.path.join(args.log_dir, 'finetune')
    finetune_args.save_dir = finetune_save_dir
    finetune_args.num_epochs = args.finetune_epochs
    
    # Track losses for this phase
    losses = {
        'total': AverageMeter(),
        'base_total': AverageMeter(),
        'refinement_total': AverageMeter()
    }
    
    # Train loop
    for epoch in range(start_epoch, args.finetune_epochs):
        print(f"Fine-tuning Epoch {epoch}/{args.finetune_epochs}")
        
        # Train for one epoch
        train_metrics = train_epoch(
            model=progressive_model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            epoch=epoch,
            device=device,
            scaler=scaler,
            args=finetune_args,
            writer=writer,
            log_dir=finetune_args.log_dir
        )
        
        # Update learning rate
        scheduler.step()
        
        # Validate
        if (epoch + 1) % args.val_interval == 0:
            val_metrics = validate(
                model=progressive_model,
                val_loader=val_loader,
                criterion=criterion,
                epoch=epoch,
                device=device,
                args=finetune_args,
                writer=writer,
                log_dir=finetune_args.log_dir
            )
            
            # Save best model
            if val_metrics['psnr'] > best_psnr:
                best_psnr = val_metrics['psnr']
                save_checkpoint(
                    model=progressive_model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    metrics=val_metrics,
                    best_psnr=best_psnr,
                    path=os.path.join(finetune_save_dir, 'best_finetune_model.pth')
                )
                print(f"Saved best fine-tuned model with PSNR {best_psnr:.2f}")
        
        # Save checkpoint periodically
        if (epoch + 1) % args.save_interval == 0:
            save_checkpoint(
                model=progressive_model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                metrics=train_metrics,
                best_psnr=best_psnr,
                path=os.path.join(finetune_save_dir, f'finetune_checkpoint_epoch_{epoch}.pth')
            )
            print(f"Saved fine-tuning checkpoint for epoch {epoch}")
    
    # Save final model
    save_checkpoint(
        model=progressive_model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=args.finetune_epochs - 1,
        metrics=train_metrics,
        best_psnr=best_psnr,
        path=os.path.join(finetune_save_dir, 'final_finetune_model.pth')
    )
    print(f"Fine-tuning completed. Final model saved.")
    
    # Close tensorboard writer
    writer.close()
    
    return progressive_model


def train_progressive_restoration(base_model_path, train_loader, val_loader, device, args):
    """Complete two-stage restoration training pipeline:
    1. Load pre-trained base model
    2. Train refinement stage with base model frozen
    3. Optional: Fine-tune entire model end-to-end
    """
    print(f"Starting two-stage restoration training pipeline...")
    
    # Step 1: Load pre-trained base model
    print(f"Loading base model from {base_model_path}...")
    base_model = DGMHRN(
        in_channels=3,
        base_channels=args.base_channels,
        num_downs=args.num_downs,
        num_degradation_types=args.num_degradation_types,
        norm_type=args.norm_type,
        activation=args.activation,
        fusion_type=args.fusion_type
    ).to(device)
    
    checkpoint = load_checkpoint(base_model_path, base_model)
    print(f"Base model loaded with PSNR: {checkpoint.get('best_psnr', 0):.2f}")
    
    # Validate base model performance
    base_criterion = CombinedLoss(
        lambda_l1=args.lambda_l1,
        lambda_ssim=args.lambda_ssim,
        lambda_perceptual=args.lambda_perceptual,
        lambda_diversity=args.lambda_diversity,
        lambda_degradation=args.lambda_degradation
    ).to(device)
    
    print("Validating base model performance...")
    base_metrics = validate(
        model=base_model,
        val_loader=val_loader,
        criterion=base_criterion,
        epoch=0,
        device=device,
        args=args
    )
    print(f"Base model validation: PSNR: {base_metrics['psnr']:.2f}, SSIM: {base_metrics['ssim']:.4f}")
    
    # Step 2: Train refinement stage with base model frozen
    print("\nTraining refinement stage with base model frozen...")
    progressive_model = train_refinement(base_model, train_loader, val_loader, device, args)
    
    # Step 3: Optional fine-tuning
    if args.finetune_epochs > 0:
        print("\nFine-tuning the entire progressive model end-to-end...")
        progressive_model = finetune_progressive_model(progressive_model, train_loader, val_loader, device, args)
    
    return progressive_model
