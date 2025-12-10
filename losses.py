import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from scipy import signal


class L1Loss(nn.Module):
    """Simple L1 loss for image reconstruction with NaN handling"""
    def __init__(self):
        super(L1Loss, self).__init__()
        self.loss = nn.L1Loss()
        
    def forward(self, pred, target):
        # Handle NaN values
        if torch.isnan(pred).any() or torch.isnan(target).any():
            pred = torch.nan_to_num(pred, nan=0.0)
            target = torch.nan_to_num(target, nan=0.0)
        
        # Ensure inputs are in valid range
        pred = torch.clamp(pred, 0.0, 1.0)
        target = torch.clamp(target, 0.0, 1.0)
            
        return self.loss(pred, target)


class SSIMLoss(nn.Module):
    """Structural Similarity Index (SSIM) loss with enhanced stability"""
    def __init__(self, window_size=11, size_average=True, channel=3):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel
        self.register_buffer('window', self._create_window(window_size, channel))
        
    def _create_window(self, window_size, channel):
        # Generate a 1D Gaussian kernel
        def gaussian(window_size, sigma):
            gauss = torch.Tensor([np.exp(-(x - window_size//2)**2/(2 * sigma**2)) for x in range(window_size)])
            return gauss/gauss.sum()
        
        # Create window
        _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window
        
    def forward(self, img1, img2):
        # Handle NaN values
        if torch.isnan(img1).any() or torch.isnan(img2).any():
            img1 = torch.nan_to_num(img1, nan=0.0)
            img2 = torch.nan_to_num(img2, nan=0.0)
            
        # Ensure inputs are in valid range
        img1 = torch.clamp(img1, 0.0, 1.0)
        img2 = torch.clamp(img2, 0.0, 1.0)
            
        # Register the window as a buffer if not already done
        if self.window.device != img1.device:
            self.window = self.window.to(img1.device)
            
        # Compute SSIM with added epsilon values to prevent division by zero
        C1 = (0.01**2) + 1e-6  # Added small epsilon
        C2 = (0.03**2) + 1e-6  # Added small epsilon
        
        try:
            mu1 = F.conv2d(img1, self.window, padding=self.window_size//2, groups=self.channel)
            mu2 = F.conv2d(img2, self.window, padding=self.window_size//2, groups=self.channel)
            
            mu1_sq = mu1.pow(2)
            mu2_sq = mu2.pow(2)
            mu1_mu2 = mu1 * mu2
            
            sigma1_sq = F.conv2d(img1 * img1, self.window, padding=self.window_size//2, groups=self.channel) - mu1_sq
            sigma2_sq = F.conv2d(img2 * img2, self.window, padding=self.window_size//2, groups=self.channel) - mu2_sq
            sigma12 = F.conv2d(img1 * img2, self.window, padding=self.window_size//2, groups=self.channel) - mu1_mu2
            
            # Add small epsilon to avoid numerical instability
            sigma1_sq = torch.clamp(sigma1_sq, min=1e-6, max=1e6)
            sigma2_sq = torch.clamp(sigma2_sq, min=1e-6, max=1e6)
            
            ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
            
            # Handle any remaining NaN values that might occur
            ssim_map = torch.nan_to_num(ssim_map, nan=0.0)
            
            if self.size_average:
                return 1 - ssim_map.mean()
            else:
                return 1 - ssim_map.mean(1).mean(1).mean(1)
                
        except Exception as e:
            print(f"Error in SSIM calculation: {e}")
            # Return a stable fallback value
            return torch.tensor(0.5, device=img1.device, requires_grad=True)


class PerceptualLoss(nn.Module):
    """Perceptual loss using VGG features with enhanced stability"""
    def __init__(self, layers=[2, 7, 12, 21, 30]):
        super(PerceptualLoss, self).__init__()
        self.layers = layers
        self.criterion = nn.L1Loss()
        self.vgg = None  # Lazy loading to save memory
        self.is_vgg_loaded = False
        
    def _load_vgg_model(self):
        # Load pretrained VGG19 model
        vgg = models.vgg19(pretrained=True).features
        vgg.eval()
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg
        self.is_vgg_loaded = True
        
    def _get_features(self, x, layers):
        features = []
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if i in layers:
                features.append(x)
        return features
        
    def forward(self, pred, target):
        # Handle NaN values
        if torch.isnan(pred).any() or torch.isnan(target).any():
            pred = torch.nan_to_num(pred, nan=0.0)
            target = torch.nan_to_num(target, nan=0.0)
            
        # Ensure inputs are in valid range
        pred = torch.clamp(pred, 0.0, 1.0)
        target = torch.clamp(target, 0.0, 1.0)
            
        try:
            # Lazy loading of VGG model
            if not self.is_vgg_loaded:
                self._load_vgg_model()
                # Move VGG to the same device as inputs
                self.vgg = self.vgg.to(pred.device)
            
            # Normalize images for VGG
            mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(pred.device)
            std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(pred.device)
            
            # Prevent division by zero with small epsilon
            pred_norm = (pred - mean) / (std + 1e-6)
            target_norm = (target - mean) / (std + 1e-6)
            
            # Get features
            pred_features = self._get_features(pred_norm, self.layers)
            target_features = self._get_features(target_norm, self.layers)
            
            # Compute loss
            loss = 0
            for i in range(len(pred_features)):
                # Handle any NaN values in features
                if torch.isnan(pred_features[i]).any() or torch.isnan(target_features[i]).any():
                    pred_features[i] = torch.nan_to_num(pred_features[i], nan=0.0)
                    target_features[i] = torch.nan_to_num(target_features[i], nan=0.0)
                    
                layer_loss = self.criterion(pred_features[i], target_features[i])
                if torch.isnan(layer_loss).any() or torch.isinf(layer_loss).any():
                    continue
                loss += layer_loss
                
            # Handle NaN in final loss
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                return torch.tensor(0.1, device=pred.device, requires_grad=True)
                
            return loss
            
        except Exception as e:
            print(f"Error in perceptual loss: {e}")
            # Return a small constant loss if an error occurs
            return torch.tensor(0.1, device=pred.device, requires_grad=True)


class SimplifiedPerceptualLoss(nn.Module):
    """A more lightweight and stable perceptual loss for ablation studies"""
    def __init__(self, base_channels=32, num_layers=3):
        super(SimplifiedPerceptualLoss, self).__init__()
        self.num_layers = num_layers
        
        # Create a simple encoder
        self.feature_layers = nn.ModuleList()
        in_channels = 3
        out_channels = base_channels
        
        for i in range(num_layers):
            layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=2),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2, inplace=True)
            )
            self.feature_layers.append(layer)
            in_channels = out_channels
            out_channels = min(out_channels * 2, 128)  # Cap at 128 channels
            
        self.criterion = nn.L1Loss()
        
    def forward(self, pred, target):
        # Handle NaN values
        if torch.isnan(pred).any() or torch.isnan(target).any():
            pred = torch.nan_to_num(pred, nan=0.0)
            target = torch.nan_to_num(target, nan=0.0)
            
        # Ensure inputs are in valid range
        pred = torch.clamp(pred, 0.0, 1.0)
        target = torch.clamp(target, 0.0, 1.0)
        
        try:
            # Extract features at multiple levels
            pred_features = []
            target_features = []
            
            p, t = pred, target
            
            for layer in self.feature_layers:
                p = layer(p)
                t = layer(t)
                
                # Handle any NaN values
                if torch.isnan(p).any():
                    p = torch.nan_to_num(p, nan=0.0)
                if torch.isnan(t).any():
                    t = torch.nan_to_num(t, nan=0.0)
                    
                pred_features.append(p)
                target_features.append(t)
            
            # Compute loss at each level
            loss = 0
            for p_feat, t_feat in zip(pred_features, target_features):
                layer_loss = self.criterion(p_feat, t_feat)
                
                # Skip NaN/Inf losses
                if torch.isnan(layer_loss).any() or torch.isinf(layer_loss).any():
                    continue
                    
                loss += layer_loss
                
            # Handle NaN in final loss
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                return torch.tensor(0.1, device=pred.device, requires_grad=True)
                
            return loss
            
        except Exception as e:
            print(f"Error in simplified perceptual loss: {e}")
            # Return a small constant loss if an error occurs
            return torch.tensor(0.1, device=pred.device, requires_grad=True)


class DiversityLoss(nn.Module):
    """Loss to encourage diversity between hypotheses with enhanced stability"""
    def __init__(self, lambda_div=0.5):
        super(DiversityLoss, self).__init__()
        self.lambda_div = lambda_div
        
    def forward(self, hypotheses):
        # Handle case with only one hypothesis
        if len(hypotheses) <= 1:
            return torch.tensor(0.0, device=hypotheses[0].device, requires_grad=True)
            
        n = len(hypotheses)
        loss = 0
        
        # Check for NaN values in hypotheses
        for i, hyp in enumerate(hypotheses):
            if torch.isnan(hyp).any():
                hypotheses[i] = torch.nan_to_num(hyp, nan=0.0)
                hypotheses[i] = torch.clamp(hypotheses[i], 0.0, 1.0)
        
        try:
            # Compute pairwise distances between hypotheses
            valid_pairs = 0
            for i in range(n):
                for j in range(i+1, n):
                    # Flatten tensors for cosine similarity
                    hyp_i_flat = hypotheses[i].view(hypotheses[i].size(0), -1)
                    hyp_j_flat = hypotheses[j].view(hypotheses[j].size(0), -1)
                    
                    # Add small epsilon to prevent numerical instability
                    norm_i = torch.norm(hyp_i_flat, dim=1, keepdim=True) + 1e-6
                    norm_j = torch.norm(hyp_j_flat, dim=1, keepdim=True) + 1e-6
                    
                    # Normalize vectors
                    hyp_i_norm = hyp_i_flat / norm_i
                    hyp_j_norm = hyp_j_flat / norm_j
                    
                    # Compute cosine similarity
                    similarity = torch.sum(hyp_i_norm * hyp_j_norm, dim=1)
                    
                    # Handle any NaN values in similarity
                    if torch.isnan(similarity).any():
                        similarity = torch.nan_to_num(similarity, nan=0.0)
                    
                    pair_loss = similarity.mean()
                    
                    # Skip NaN/Inf losses
                    if torch.isnan(pair_loss).any() or torch.isinf(pair_loss).any():
                        continue
                        
                    loss += pair_loss
                    valid_pairs += 1
            
            # Normalize by number of valid pairs
            if valid_pairs > 0:
                loss = loss / valid_pairs
            else:
                # If no valid pairs, return a zero tensor with gradient
                return torch.tensor(0.0, device=hypotheses[0].device, requires_grad=True)
            
            # Handle NaN in final loss
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                return torch.tensor(0.0, device=hypotheses[0].device, requires_grad=True)
                
            return self.lambda_div * loss
            
        except Exception as e:
            print(f"Error in diversity loss: {e}")
            return torch.tensor(0.0, device=hypotheses[0].device, requires_grad=True)


class DegradationConsistencyLoss(nn.Module):
    """Loss to ensure degradation maps are consistent with actual degradations"""
    def __init__(self, lambda_consist=0.1):
        super(DegradationConsistencyLoss, self).__init__()
        self.lambda_consist = lambda_consist
        self.mse = nn.MSELoss()
        
    def forward(self, degradation_maps, degraded_img, reference_img):
        # Handle NaN values
        if torch.isnan(degradation_maps).any() or torch.isnan(degraded_img).any() or torch.isnan(reference_img).any():
            degradation_maps = torch.nan_to_num(degradation_maps, nan=0.0)
            degraded_img = torch.nan_to_num(degraded_img, nan=0.0)
            reference_img = torch.nan_to_num(reference_img, nan=0.0)
            
        # Ensure inputs are in valid range
        degraded_img = torch.clamp(degraded_img, 0.0, 1.0)
        reference_img = torch.clamp(reference_img, 0.0, 1.0)
        degradation_maps = torch.clamp(degradation_maps, 0.0, 1.0)
            
        try:
            # Compute difference map between degraded and reference
            diff = torch.abs(degraded_img - reference_img)
            
            # Simplify by using the mean across channels
            diff_mean = diff.mean(dim=1, keepdim=True)
            
            # Get total degradation by summing all maps
            total_degradation = degradation_maps.sum(dim=1, keepdim=True)
            
            # Normalize both maps to [0,1] range for comparison with epsilon to prevent division by zero
            diff_max = diff_mean.max() + 1e-6
            total_max = total_degradation.max() + 1e-6
            
            diff_mean = diff_mean / diff_max
            total_degradation = total_degradation / total_max
            
            # Compute consistency loss
            loss = self.mse(total_degradation, diff_mean)
            
            # Handle any NaN values in loss
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                return torch.tensor(0.1, device=degradation_maps.device, requires_grad=True)
                
            return self.lambda_consist * loss
            
        except Exception as e:
            print(f"Error in degradation consistency loss: {e}")
            return torch.tensor(0.1, device=degradation_maps.device, requires_grad=True)


class CombinedLoss(nn.Module):
    """Combined loss function for training the DGMHRN model with enhanced stability"""
    def __init__(self, lambda_l1=1.0, lambda_ssim=0.1, lambda_perceptual=0.1, 
                 lambda_diversity=0.05, lambda_degradation=0.1, use_simplified_perceptual=True):
        super(CombinedLoss, self).__init__()
        self.l1_loss = L1Loss()
        self.ssim_loss = SSIMLoss()
        
        # Use simplified perceptual loss for better stability
        if use_simplified_perceptual:
            self.perceptual_loss = SimplifiedPerceptualLoss()
        else:
            self.perceptual_loss = PerceptualLoss()
            
        self.diversity_loss = DiversityLoss(lambda_div=1.0)  # Will be scaled by lambda_diversity
        self.degradation_loss = DegradationConsistencyLoss(lambda_consist=1.0)  # Will be scaled by lambda_degradation
        
        self.lambda_l1 = lambda_l1
        self.lambda_ssim = lambda_ssim
        self.lambda_perceptual = lambda_perceptual
        self.lambda_diversity = lambda_diversity
        self.lambda_degradation = lambda_degradation
        
    def forward(self, outputs, targets, degraded_img):
        """
        Args:
            outputs: Dict containing 'restored_image', 'degradation_maps', and 'hypotheses'
            targets: Ground truth reference images
            degraded_img: Original degraded input images
        """
        # Handle NaN values in inputs
        if torch.isnan(targets).any():
            targets = torch.nan_to_num(targets, nan=0.0)
            targets = torch.clamp(targets, 0.0, 1.0)
            
        if torch.isnan(degraded_img).any():
            degraded_img = torch.nan_to_num(degraded_img, nan=0.0)
            degraded_img = torch.clamp(degraded_img, 0.0, 1.0)
            
        # Get outputs
        restored_img = outputs['restored_image']
        degradation_maps = outputs['degradation_maps']
        hypotheses = outputs['hypotheses']
        
        # Handle NaN values in outputs
        if torch.isnan(restored_img).any():
            restored_img = torch.nan_to_num(restored_img, nan=0.0)
            restored_img = torch.clamp(restored_img, 0.0, 1.0)
            
        if torch.isnan(degradation_maps).any():
            degradation_maps = torch.nan_to_num(degradation_maps, nan=0.0)
            degradation_maps = torch.clamp(degradation_maps, 0.0, 1.0)
        
        # Compute individual loss components with try/except for each
        try:
            l1 = self.l1_loss(restored_img, targets)
            if torch.isnan(l1).any() or torch.isinf(l1).any():
                print("Warning: NaN/Inf in L1 loss, using fallback")
                l1 = torch.tensor(0.1, device=restored_img.device, requires_grad=True)
        except Exception as e:
            print(f"Error computing L1 loss: {e}")
            l1 = torch.tensor(0.1, device=restored_img.device, requires_grad=True)
            
        try:
            ssim = self.ssim_loss(restored_img, targets)
            if torch.isnan(ssim).any() or torch.isinf(ssim).any():
                print("Warning: NaN/Inf in SSIM loss, using fallback")
                ssim = torch.tensor(0.1, device=restored_img.device, requires_grad=True)
        except Exception as e:
            print(f"Error computing SSIM loss: {e}")
            ssim = torch.tensor(0.1, device=restored_img.device, requires_grad=True)
        
        # Compute perceptual loss
        if self.lambda_perceptual > 0:
            try:
                perceptual = self.perceptual_loss(restored_img, targets)
                if torch.isnan(perceptual).any() or torch.isinf(perceptual).any():
                    print("Warning: NaN/Inf in perceptual loss, using fallback")
                    perceptual = torch.tensor(0.1, device=restored_img.device, requires_grad=True)
            except Exception as e:
                print(f"Error computing perceptual loss: {e}")
                perceptual = torch.tensor(0.1, device=restored_img.device, requires_grad=True)
        else:
            perceptual = torch.tensor(0.0, device=restored_img.device)
        
        # Compute diversity loss
        if self.lambda_diversity > 0 and len(hypotheses) > 1:
            try:
                diversity = self.diversity_loss(hypotheses)
                if torch.isnan(diversity).any() or torch.isinf(diversity).any():
                    print("Warning: NaN/Inf in diversity loss, using fallback")
                    diversity = torch.tensor(0.0, device=restored_img.device, requires_grad=True)
            except Exception as e:
                print(f"Error computing diversity loss: {e}")
                diversity = torch.tensor(0.0, device=restored_img.device, requires_grad=True)
        else:
            diversity = torch.tensor(0.0, device=restored_img.device)
        
        # Compute degradation consistency loss
        if self.lambda_degradation > 0:
            try:
                degradation = self.degradation_loss(degradation_maps, degraded_img, targets)
                if torch.isnan(degradation).any() or torch.isinf(degradation).any():
                    print("Warning: NaN/Inf in degradation loss, using fallback")
                    degradation = torch.tensor(0.1, device=restored_img.device, requires_grad=True)
            except Exception as e:
                print(f"Error computing degradation loss: {e}")
                degradation = torch.tensor(0.1, device=restored_img.device, requires_grad=True)
        else:
            degradation = torch.tensor(0.0, device=restored_img.device)
        
        # Compute individual hypothesis losses
        try:
            hypotheses_loss = 0
            valid_hyp_count = 0
            
            for hypothesis in hypotheses:
                # Handle NaN values
                if torch.isnan(hypothesis).any():
                    hypothesis = torch.nan_to_num(hypothesis, nan=0.0)
                    hypothesis = torch.clamp(hypothesis, 0.0, 1.0)
                
                hyp_loss = self.l1_loss(hypothesis, targets)
                
                # Skip NaN/Inf losses
                if torch.isnan(hyp_loss).any() or torch.isinf(hyp_loss).any():
                    continue
                    
                hypotheses_loss += hyp_loss
                valid_hyp_count += 1
                
            if valid_hyp_count > 0:
                hypotheses_loss = hypotheses_loss / valid_hyp_count
            else:
                print("Warning: No valid hypothesis losses, using fallback")
                hypotheses_loss = torch.tensor(0.1, device=restored_img.device, requires_grad=True)
                
        except Exception as e:
            print(f"Error computing hypothesis losses: {e}")
            hypotheses_loss = torch.tensor(0.1, device=restored_img.device, requires_grad=True)
        
        # Combine all loss components with their weights
        try:
            total_loss = (
                self.lambda_l1 * l1 + 
                self.lambda_ssim * ssim + 
                self.lambda_perceptual * perceptual + 
                self.lambda_diversity * diversity + 
                self.lambda_degradation * degradation +
                0.5 * hypotheses_loss  # Auxiliary loss for individual hypotheses
            )
            
            # Final check for NaN in total loss
            if torch.isnan(total_loss).any() or torch.isinf(total_loss).any():
                print("NaN/Inf detected in total loss! Using fallback value.")
                total_loss = torch.tensor(1.0, device=restored_img.device, requires_grad=True)
                
        except Exception as e:
            print(f"Error computing total loss: {e}")
            total_loss = torch.tensor(1.0, device=restored_img.device, requires_grad=True)
        
        # Return total loss and individual components for logging
        return {
            'total': total_loss,
            'l1': l1,
            'ssim': ssim,
            'perceptual': perceptual,
            'diversity': diversity,
            'degradation': degradation,
            'hypotheses': hypotheses_loss
        }


class RefinementLoss(nn.Module):
    """Loss function for training the progressive refinement stage with enhanced stability"""
    def __init__(self, lambda_recon=1.0, lambda_magnitude=0.1, lambda_improve=0.5, 
                 lambda_ssim=0.1, lambda_perceptual=0.1, use_simplified_perceptual=True):
        super(RefinementLoss, self).__init__()
        self.l1_loss = L1Loss()
        self.ssim_loss = SSIMLoss()
        
        # Use simplified perceptual loss for better stability
        if use_simplified_perceptual:
            self.perceptual_loss = SimplifiedPerceptualLoss()
        else:
            self.perceptual_loss = PerceptualLoss()
        
        self.lambda_recon = lambda_recon
        self.lambda_magnitude = lambda_magnitude
        self.lambda_improve = lambda_improve
        self.lambda_ssim = lambda_ssim
        self.lambda_perceptual = lambda_perceptual
        
    def forward(self, outputs, targets, degraded_img=None):
        """
        Args:
            outputs: Dict containing 'restored_image', 'initial_restoration', 'refinement', etc.
            targets: Ground truth reference images
            degraded_img: Original degraded input images (optional)
        """
        # Handle NaN values
        if torch.isnan(targets).any():
            targets = torch.nan_to_num(targets, nan=0.0)
            targets = torch.clamp(targets, 0.0, 1.0)
            
        if degraded_img is not None and torch.isnan(degraded_img).any():
            degraded_img = torch.nan_to_num(degraded_img, nan=0.0)
            degraded_img = torch.clamp(degraded_img, 0.0, 1.0)
        
        # Get outputs
        try:
            initial_restoration = outputs['initial_restoration']
            final_restoration = outputs['restored_image']
            refinement = outputs['refinement']
            
            # Handle NaN values
            if torch.isnan(initial_restoration).any():
                initial_restoration = torch.nan_to_num(initial_restoration, nan=0.0)
                initial_restoration = torch.clamp(initial_restoration, 0.0, 1.0)
                
            if torch.isnan(final_restoration).any():
                final_restoration = torch.nan_to_num(final_restoration, nan=0.0)
                final_restoration = torch.clamp(final_restoration, 0.0, 1.0)
                
            if torch.isnan(refinement).any():
                refinement = torch.nan_to_num(refinement, nan=0.0)
        except KeyError as e:
            print(f"Missing key in outputs: {e}")
            # Return a fallback loss
            return {
                'total': torch.tensor(1.0, device=targets.device, requires_grad=True),
                'recon': torch.tensor(0.5, device=targets.device),
                'magnitude': torch.tensor(0.1, device=targets.device),
                'improvement': torch.tensor(0.4, device=targets.device)
            }
        
        # 1. Reconstruction loss for final output
        try:
            l1_final = self.l1_loss(final_restoration, targets)
            if torch.isnan(l1_final).any() or torch.isinf(l1_final).any():
                print("Warning: NaN/Inf in final L1 loss, using fallback")
                l1_final = torch.tensor(0.2, device=final_restoration.device, requires_grad=True)
        except Exception as e:
            print(f"Error computing final L1 loss: {e}")
            l1_final = torch.tensor(0.2, device=final_restoration.device, requires_grad=True)
            
        try:
            ssim_final = self.ssim_loss(final_restoration, targets)
            if torch.isnan(ssim_final).any() or torch.isinf(ssim_final).any():
                print("Warning: NaN/Inf in final SSIM loss, using fallback")
                ssim_final = torch.tensor(0.1, device=final_restoration.device, requires_grad=True)
        except Exception as e:
            print(f"Error computing final SSIM loss: {e}")
            ssim_final = torch.tensor(0.1, device=final_restoration.device, requires_grad=True)
            
        try:
            perceptual_final = self.perceptual_loss(final_restoration, targets)
            if torch.isnan(perceptual_final).any() or torch.isinf(perceptual_final).any():
                print("Warning: NaN/Inf in final perceptual loss, using fallback")
                perceptual_final = torch.tensor(0.2, device=final_restoration.device, requires_grad=True)
        except Exception as e:
            print(f"Error computing final perceptual loss: {e}")
            perceptual_final = torch.tensor(0.2, device=final_restoration.device, requires_grad=True)
        
        # Combine reconstruction losses
        recon_loss = l1_final + self.lambda_ssim * ssim_final + self.lambda_perceptual * perceptual_final
        
        # 2. Refinement magnitude loss (encourage minimal refinement)
        try:
            magnitude_loss = torch.mean(torch.abs(refinement))
            if torch.isnan(magnitude_loss).any() or torch.isinf(magnitude_loss).any():
                print("Warning: NaN/Inf in magnitude loss, using fallback")
                magnitude_loss = torch.tensor(0.1, device=final_restoration.device, requires_grad=True)
        except Exception as e:
            print(f"Error computing magnitude loss: {e}")
            magnitude_loss = torch.tensor(0.1, device=final_restoration.device, requires_grad=True)
        
        # 3. Progressive improvement loss
        try:
            # Calculate L1 error for initial and final restoration J_hats
            l1_initial = self.l1_loss(initial_restoration, targets)
            
            # The final restoration J_hat should have lower error than initial
            # We use ReLU to only penalize when final error is higher than initial
            # The 0.01 term ensures we want at least a small improvement
            improvement_loss = F.relu(l1_final - l1_initial + 0.01)
            
            # Calculate improvement in perceptual and SSIM metrics
            ssim_initial = self.ssim_loss(initial_restoration, targets)
            perceptual_initial = self.perceptual_loss(initial_restoration, targets)
            
            ssim_improve = F.relu(ssim_final - ssim_initial + 0.005)
            perceptual_improve = F.relu(perceptual_final - perceptual_initial + 0.005)
            
            # Combined improvement loss
            combined_improve = improvement_loss + 0.2 * (ssim_improve + perceptual_improve)
            
            # Handle NaN/Inf
            if torch.isnan(combined_improve).any() or torch.isinf(combined_improve).any():
                print("Warning: NaN/Inf in improvement loss, using fallback")
                combined_improve = torch.tensor(0.1, device=final_restoration.device, requires_grad=True)
                l1_improve = torch.tensor(0.05, device=final_restoration.device)
                ssim_improve = torch.tensor(0.05, device=final_restoration.device)
                perceptual_improve = torch.tensor(0.05, device=final_restoration.device)
            else:
                l1_improve = l1_initial - l1_final
                
        except Exception as e:
            print(f"Error computing improvement loss: {e}")
            combined_improve = torch.tensor(0.1, device=final_restoration.device, requires_grad=True)
            l1_improve = torch.tensor(0.05, device=final_restoration.device)
            ssim_improve = torch.tensor(0.05, device=final_restoration.device)
            perceptual_improve = torch.tensor(0.05, device=final_restoration.device)
        
        # Combined loss
        try:
            total_loss = (
                self.lambda_recon * recon_loss +
                self.lambda_magnitude * magnitude_loss +
                self.lambda_improve * combined_improve
            )
            
            # Check for NaN in total loss
            if torch.isnan(total_loss).any() or torch.isinf(total_loss).any():
                print("NaN/Inf detected in refinement total loss! Using fallback.")
                total_loss = torch.tensor(1.0, device=final_restoration.device, requires_grad=True)
                
        except Exception as e:
            print(f"Error computing refinement total loss: {e}")
            total_loss = torch.tensor(1.0, device=final_restoration.device, requires_grad=True)
        
        # Return total loss and individual components for logging
        return {
            'total': total_loss,
            'recon': recon_loss,
            'l1_final': l1_final,
            'ssim_final': ssim_final,
            'perceptual_final': perceptual_final,
            'magnitude': magnitude_loss,
            'improvement': combined_improve,
            'l1_improve': l1_improve,
            'ssim_improve': ssim_improve,
            'perceptual_improve': perceptual_improve
        }


class ProgressiveCombinedLoss(nn.Module):
    """Combined loss for end-to-end training of the two-stage restoration network"""
    def __init__(self, lambda_base=1.0, lambda_refinement=1.0, base_loss_weights=None, 
                 refinement_loss_weights=None, use_simplified_perceptual=True):
        super(ProgressiveCombinedLoss, self).__init__()
        
        # Default weights for base model loss
        if base_loss_weights is None:
            base_loss_weights = {
                'lambda_l1': 1.0,
                'lambda_ssim': 0.1,
                'lambda_perceptual': 0.1,
                'lambda_diversity': 0.05,
                'lambda_degradation': 0.1,
                'use_simplified_perceptual': use_simplified_perceptual
            }
        else:
            base_loss_weights['use_simplified_perceptual'] = use_simplified_perceptual
        
        # Default weights for refinement loss
        if refinement_loss_weights is None:
            refinement_loss_weights = {
                'lambda_recon': 1.0,
                'lambda_magnitude': 0.1,
                'lambda_improve': 0.5,
                'lambda_ssim': 0.1,
                'lambda_perceptual': 0.1,
                'use_simplified_perceptual': use_simplified_perceptual
            }
        else:
            refinement_loss_weights['use_simplified_perceptual'] = use_simplified_perceptual
        
        # Create the individual loss modules
        self.base_loss = CombinedLoss(**base_loss_weights)
        self.refinement_loss = RefinementLoss(**refinement_loss_weights)
        
        # Weights for combining the losses
        self.lambda_base = lambda_base
        self.lambda_refinement = lambda_refinement
        
    def forward(self, outputs, targets, degraded_img):
        """
        Args:
            outputs: Dict containing all outputs from the progressive model
            targets: Ground truth reference images
            degraded_img: Original degraded input images
        """
        # Handle NaN values
        if torch.isnan(targets).any():
            targets = torch.nan_to_num(targets, nan=0.0)
            targets = torch.clamp(targets, 0.0, 1.0)
            
        if torch.isnan(degraded_img).any():
            degraded_img = torch.nan_to_num(degraded_img, nan=0.0)
            degraded_img = torch.clamp(degraded_img, 0.0, 1.0)
        
        try:
            # Create a subset of outputs for the base model
            base_outputs = {
                'restored_image': outputs['initial_restoration'],
                'degradation_maps': outputs['degradation_maps'],
                'hypotheses': outputs['hypotheses']
            }
            
            # Calculate base model loss
            base_loss_dict = self.base_loss(base_outputs, targets, degraded_img)
            
            # Calculate refinement loss
            refinement_loss_dict = self.refinement_loss(outputs, targets, degraded_img)
            
            # Combined total loss
            total_loss = (
                self.lambda_base * base_loss_dict['total'] +
                self.lambda_refinement * refinement_loss_dict['total']
            )
            
            # Check for NaN in total loss
            if torch.isnan(total_loss).any() or torch.isinf(total_loss).any():
                print("NaN detected in progressive combined loss! Using fallback.")
                total_loss = torch.tensor(1.0, device=targets.device, requires_grad=True)
                base_loss_dict['total'] = torch.tensor(0.5, device=targets.device)
                refinement_loss_dict['total'] = torch.tensor(0.5, device=targets.device)
                
        except Exception as e:
            print(f"Error in progressive combined loss: {e}")
            return {
                'total': torch.tensor(1.0, device=targets.device, requires_grad=True),
                'base_total': torch.tensor(0.5, device=targets.device),
                'refinement_total': torch.tensor(0.5, device=targets.device)
            }
        
        # Combine loss dictionaries for logging
        combined_dict = {
            'total': total_loss,
            'base_total': base_loss_dict['total'],
            'refinement_total': refinement_loss_dict['total']
        }
        
        # Add base loss components with 'base_' prefix
        for k, v in base_loss_dict.items():
            if k != 'total':
                combined_dict[f'base_{k}'] = v
        
        # Add refinement loss components with 'ref_' prefix
        for k, v in refinement_loss_dict.items():
            if k != 'total':
                combined_dict[f'ref_{k}'] = v
        
        return combined_dict
    