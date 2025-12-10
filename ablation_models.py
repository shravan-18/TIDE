import torch
import torch.nn as nn
import torch.nn.functional as F
from model import DGMHRN, TIDE, RestorationDecoder
from model import Encoder, ColorRestorationDecoder, ContrastEnhancementDecoder, DetailRecoveryDecoder, DenoisingDecoder
import numpy as np


class StableFusion(nn.Module):
    """Stable fusion module to replace direct averaging"""
    def __init__(self, num_hypotheses):
        super(StableFusion, self).__init__()
        self.num_hypotheses = num_hypotheses
        # Create learnable weights that sum to 1
        self.weights = nn.Parameter(torch.ones(num_hypotheses) / num_hypotheses)
        
    def forward(self, hypotheses):
        # Ensure weights sum to 1 using softmax
        norm_weights = F.softmax(self.weights, dim=0)
        
        # Apply weights to each hypothesis
        fused = None
        for i, hyp in enumerate(hypotheses):
            if fused is None:
                fused = norm_weights[i] * hyp
            else:
                fused = fused + norm_weights[i] * hyp
                
        return fused


class NoDegradationMapsDGMHRN(nn.Module):
    """Ablation model without degradation maps, using stable fusion mechanism"""
    def __init__(self, in_channels=3, base_channels=64, num_downs=5, 
                 norm_type='instance', activation='leaky_relu', **kwargs):
        super(NoDegradationMapsDGMHRN, self).__init__()
        
        # Create a complete base model
        self.base_model = DGMHRN(
            in_channels=in_channels,
            base_channels=base_channels,
            num_downs=num_downs,
            num_degradation_types=4,  # Keep the normal number of degradations
            norm_type=norm_type,
            activation=activation,
            fusion_type='learned'
        )
        
        # Replace the fusion module with our stable version
        self.stable_fusion = StableFusion(num_hypotheses=4)
        
        # Fix model initialization with smaller weights
        self._initialize_weights()
        
        # Initialize dynamic layers with a dummy forward pass
        self._init_dynamic_layers()
        
    def _initialize_weights(self):
        """Initialize weights to ensure stable training"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # Scale down weights to prevent explosion - more aggressive scaling
                with torch.no_grad():
                    m.weight.data *= 0.3  # More aggressive scaling (was 0.5)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def _init_dynamic_layers(self):
        """Initialize any dynamic layers with a dummy forward pass"""
        if next(self.parameters()).is_cuda:
            device = next(self.parameters()).device
            dummy_input = torch.zeros(1, 3, 64, 64, device=device)
        else:
            dummy_input = torch.zeros(1, 3, 64, 64)
            
        # Run a dummy forward pass in eval mode to initialize all dynamic layers
        self.eval()
        with torch.no_grad():
            try:
                _ = self.forward(dummy_input)
            except Exception as e:
                print(f"Warning during dynamic layer initialization: {e}")
                # If 64x64 is too small, try a larger size
                try:
                    dummy_input = torch.zeros(1, 3, 128, 128, device=dummy_input.device)
                    _ = self.forward(dummy_input)
                except Exception as e:
                    print(f"Failed to initialize dynamic layers: {e}")
        self.train()  # Set back to training mode
        
    def forward(self, x):
        # Validate input
        if torch.isnan(x).any():
            x = torch.nan_to_num(x, nan=0.0)
        
        # Extract features
        encoder_features = self.base_model.feature_extractor(x)
        
        # Ensure encoder features don't have NaNs
        for i in range(len(encoder_features)):
            if torch.isnan(encoder_features[i]).any():
                encoder_features[i] = torch.nan_to_num(encoder_features[i], nan=0.0)
        
        # Generate hypotheses
        color_hypothesis = self.base_model.color_decoder(encoder_features)
        contrast_hypothesis = self.base_model.contrast_decoder(encoder_features)
        detail_hypothesis = self.base_model.detail_decoder(encoder_features)
        denoise_hypothesis = self.base_model.denoise_decoder(encoder_features)
        
        # Ensure all hypotheses are valid and clip to [0, 1]
        hypotheses = []
        for i, hyp in enumerate([color_hypothesis, contrast_hypothesis, detail_hypothesis, denoise_hypothesis]):
            if torch.isnan(hyp).any():
                hyp = torch.nan_to_num(hyp, nan=0.0)
            hyp = torch.clamp(hyp, 0.0, 1.0)
            hypotheses.append(hyp)
        
        # Use stable fusion instead of simple averaging
        fused_image = self.stable_fusion(hypotheses)
        
        # Ensure output is in valid range and has no NaNs
        fused_image = torch.nan_to_num(fused_image, nan=0.0)
        fused_image = torch.clamp(fused_image, 0.0, 1.0)
        
        # Create dummy degradation maps with same spatial dimensions as input
        batch_size, _, height, width = x.size()
        dummy_maps = torch.ones(batch_size, 4, height, width, device=x.device) / 4.0
        
        # Return outputs in consistent format
        return {
            'restored_image': fused_image,
            'degradation_maps': dummy_maps,
            'hypotheses': hypotheses
        }


class SingleHypothesisDGMHRN(nn.Module):
    """Ablation model with only a single hypothesis (color restoration)"""
    def __init__(self, in_channels=3, base_channels=64, num_downs=5, 
                 norm_type='instance', activation='leaky_relu', **kwargs):
        super(SingleHypothesisDGMHRN, self).__init__()
        
        # Create encoder directly instead of using the base model
        self.feature_extractor = Encoder(
            in_channels=in_channels, 
            base_channels=base_channels,
            num_downs=num_downs,
            norm_type=norm_type,
            activation=activation
        )
        
        # Get encoder feature dimensions
        self.encoder_features = self.feature_extractor.features
        
        # Create only the color decoder
        self.color_decoder = ColorRestorationDecoder(
            self.encoder_features, 
            norm_type=norm_type, 
            activation=activation
        )
        
        # Initialize with smaller weights
        self._initialize_weights()
        
        # Initialize dynamic layers
        self._init_dynamic_layers()
        
    def _initialize_weights(self):
        """Initialize weights to ensure stable training"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # Scale down weights to prevent explosion
                with torch.no_grad():
                    m.weight.data *= 0.3  # More aggressive scaling
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def _init_dynamic_layers(self):
        """Initialize any dynamic layers with a dummy forward pass"""
        if next(self.parameters()).is_cuda:
            device = next(self.parameters()).device
            dummy_input = torch.zeros(1, 3, 64, 64, device=device)
        else:
            dummy_input = torch.zeros(1, 3, 64, 64)
            
        # Run a dummy forward pass in eval mode to initialize all dynamic layers
        self.eval()
        with torch.no_grad():
            try:
                _ = self.forward(dummy_input)
            except Exception as e:
                print(f"Warning during dynamic layer initialization: {e}")
                # If 64x64 is too small, try a larger size
                try:
                    dummy_input = torch.zeros(1, 3, 128, 128, device=dummy_input.device)
                    _ = self.forward(dummy_input)
                except Exception as e:
                    print(f"Failed to initialize dynamic layers: {e}")
        self.train()  # Set back to training mode
        
    def forward(self, x):
        # Validate input
        if torch.isnan(x).any():
            x = torch.nan_to_num(x, nan=0.0)
            
        # Extract features
        encoder_features = self.feature_extractor(x)
        
        # Check for NaNs in features
        for i in range(len(encoder_features)):
            if torch.isnan(encoder_features[i]).any():
                encoder_features[i] = torch.nan_to_num(encoder_features[i], nan=0.0)
        
        # Generate only color hypothesis
        color_hypothesis = self.color_decoder(encoder_features)
        
        # Ensure output is valid
        if torch.isnan(color_hypothesis).any():
            color_hypothesis = torch.nan_to_num(color_hypothesis, nan=0.0)
        color_hypothesis = torch.clamp(color_hypothesis, 0.0, 1.0)
        
        # No fusion needed for single hypothesis
        restored_image = color_hypothesis
        
        # Create dummy degradation maps for compatibility
        batch_size, _, height, width = x.size()
        dummy_maps = torch.ones(batch_size, 1, height, width, device=x.device)
        
        # Return outputs in consistent format
        return {
            'restored_image': restored_image,
            'degradation_maps': dummy_maps,
            'hypotheses': [color_hypothesis]
        }


class FusionTypeDGMHRN(nn.Module):
    """Ablation model with configurable fusion type"""
    def __init__(self, in_channels=3, base_channels=64, num_downs=5, 
                 num_degradation_types=4, norm_type='instance', activation='leaky_relu',
                 fusion_type='learned', **kwargs):
        super(FusionTypeDGMHRN, self).__init__()
        
        # Create a complete base model
        self.base_model = DGMHRN(
            in_channels=in_channels,
            base_channels=base_channels,
            num_downs=num_downs,
            num_degradation_types=num_degradation_types,
            norm_type=norm_type,
            activation=activation,
            fusion_type=fusion_type
        )
        
        # Initialize with smaller weights
        self._initialize_weights()
        
        # Initialize dynamic layers with a dummy forward pass
        self._init_dynamic_layers()
        
    def _initialize_weights(self):
        """Initialize weights to ensure stable training"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # Scale down weights to prevent explosion
                with torch.no_grad():
                    m.weight.data *= 0.3
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def _init_dynamic_layers(self):
        """Initialize any dynamic layers with a dummy forward pass"""
        if next(self.parameters()).is_cuda:
            device = next(self.parameters()).device
            dummy_input = torch.zeros(1, 3, 64, 64, device=device)
        else:
            dummy_input = torch.zeros(1, 3, 64, 64)
            
        # Run a dummy forward pass in eval mode to initialize all dynamic layers
        self.eval()
        with torch.no_grad():
            try:
                _ = self.forward(dummy_input)
            except Exception as e:
                print(f"Warning during dynamic layer initialization: {e}")
                # If 64x64 is too small, try a larger size
                try:
                    dummy_input = torch.zeros(1, 3, 128, 128, device=dummy_input.device)
                    _ = self.forward(dummy_input)
                except Exception as e:
                    print(f"Failed to initialize dynamic layers: {e}")
        self.train()  # Set back to training mode
        
    def forward(self, x):
        # Validate input
        if torch.isnan(x).any():
            x = torch.nan_to_num(x, nan=0.0)
        
        # Use the base model's forward pass directly
        outputs = self.base_model(x)
        
        # Ensure all outputs are valid
        outputs['restored_image'] = torch.clamp(outputs['restored_image'], 0.0, 1.0)
        
        # Ensure degradation maps are valid
        if torch.isnan(outputs['degradation_maps']).any():
            outputs['degradation_maps'] = torch.nan_to_num(outputs['degradation_maps'], nan=0.0)
            # Ensure maps sum to 1 along the channel dimension
            total = outputs['degradation_maps'].sum(dim=1, keepdim=True).clamp(min=1e-6)
            outputs['degradation_maps'] = outputs['degradation_maps'] / total
            
        # Ensure hypotheses are valid
        for i in range(len(outputs['hypotheses'])):
            if torch.isnan(outputs['hypotheses'][i]).any():
                outputs['hypotheses'][i] = torch.nan_to_num(outputs['hypotheses'][i], nan=0.0)
            outputs['hypotheses'][i] = torch.clamp(outputs['hypotheses'][i], 0.0, 1.0)
            
        return outputs


class DecoderCombinationDGMHRN(nn.Module):
    """Ablation model with configurable combination of decoders"""
    def __init__(self, in_channels=3, base_channels=64, num_downs=5, 
                 norm_type='instance', activation='leaky_relu',
                 fusion_type='learned', decoder_types=None, **kwargs):
        super(DecoderCombinationDGMHRN, self).__init__()
        
        if decoder_types is None:
            decoder_types = ['color', 'contrast', 'detail', 'denoise']
        
        # Set active decoders
        self.active_decoders = decoder_types
        
        # Create encoder
        self.feature_extractor = Encoder(
            in_channels=in_channels,
            base_channels=base_channels,
            num_downs=num_downs,
            norm_type=norm_type,
            activation=activation
        )
        
        # Get encoder feature dimensions
        self.encoder_features = self.feature_extractor.features
        
        # Create decoders based on specified types
        if 'color' in decoder_types:
            self.color_decoder = ColorRestorationDecoder(
                self.encoder_features, norm_type=norm_type, activation=activation
            )
            
        if 'contrast' in decoder_types:
            self.contrast_decoder = ContrastEnhancementDecoder(
                self.encoder_features, norm_type=norm_type, activation=activation
            )
            
        if 'detail' in decoder_types:
            self.detail_decoder = DetailRecoveryDecoder(
                self.encoder_features, norm_type=norm_type, activation=activation
            )
            
        if 'denoise' in decoder_types:
            self.denoise_decoder = DenoisingDecoder(
                self.encoder_features, norm_type=norm_type, activation=activation
            )
        
        # Create degradation estimator with correct number of outputs
        from model import DegradationEstimator
        self.degradation_estimator = DegradationEstimator(
            in_channels=in_channels,
            base_channels=base_channels//2,
            num_degradation_types=len(decoder_types),
            norm_type=norm_type,
            activation=activation
        )
        
        # Create fusion module
        from model import AdaptiveFusion
        self.fusion = AdaptiveFusion(
            num_hypotheses=len(decoder_types),
            num_degradation_types=len(decoder_types),
            norm_type=norm_type,
            fusion_type=fusion_type
        )
        
        # Create stable fusion as fallback
        self.stable_fusion = StableFusion(num_hypotheses=len(decoder_types))
        
        # Initialize with smaller weights
        self._initialize_weights()
        
        # Initialize dynamic layers
        self._init_dynamic_layers()
        
    def _initialize_weights(self):
        """Initialize weights to ensure stable training"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # Scale down weights to prevent explosion
                with torch.no_grad():
                    m.weight.data *= 0.3
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def _init_dynamic_layers(self):
        """Initialize any dynamic layers with a dummy forward pass"""
        if next(self.parameters()).is_cuda:
            device = next(self.parameters()).device
            dummy_input = torch.zeros(1, 3, 64, 64, device=device)
        else:
            dummy_input = torch.zeros(1, 3, 64, 64)
            
        # Run a dummy forward pass in eval mode to initialize all dynamic layers
        self.eval()
        with torch.no_grad():
            try:
                _ = self.forward(dummy_input)
            except Exception as e:
                print(f"Warning during dynamic layer initialization: {e}")
                # If 64x64 is too small, try a larger size
                try:
                    dummy_input = torch.zeros(1, 3, 128, 128, device=dummy_input.device)
                    _ = self.forward(dummy_input)
                except Exception as e:
                    print(f"Failed to initialize dynamic layers: {e}")
        self.train()  # Set back to training mode
        
    def forward(self, x):
        # Validate input
        if torch.isnan(x).any():
            x = torch.nan_to_num(x, nan=0.0)
        
        # Extract features
        encoder_features = self.feature_extractor(x)
        
        # Check for NaNs in features
        for i in range(len(encoder_features)):
            if torch.isnan(encoder_features[i]).any():
                encoder_features[i] = torch.nan_to_num(encoder_features[i], nan=0.0)
        
        # Estimate degradation maps
        degradation_maps = self.degradation_estimator(x)
        
        # Ensure degradation maps are valid
        if torch.isnan(degradation_maps).any():
            degradation_maps = torch.nan_to_num(degradation_maps, nan=0.0)
            # Ensure maps sum to 1 along the channel dimension
            total = degradation_maps.sum(dim=1, keepdim=True).clamp(min=1e-6)
            degradation_maps = degradation_maps / total
        
        # Generate only the specified hypotheses
        hypotheses = []
        if 'color' in self.active_decoders:
            color_hyp = self.color_decoder(encoder_features)
            if torch.isnan(color_hyp).any():
                color_hyp = torch.nan_to_num(color_hyp, nan=0.0)
            color_hyp = torch.clamp(color_hyp, 0.0, 1.0)
            hypotheses.append(color_hyp)
            
        if 'contrast' in self.active_decoders:
            contrast_hyp = self.contrast_decoder(encoder_features)
            if torch.isnan(contrast_hyp).any():
                contrast_hyp = torch.nan_to_num(contrast_hyp, nan=0.0)
            contrast_hyp = torch.clamp(contrast_hyp, 0.0, 1.0)
            hypotheses.append(contrast_hyp)
            
        if 'detail' in self.active_decoders:
            detail_hyp = self.detail_decoder(encoder_features)
            if torch.isnan(detail_hyp).any():
                detail_hyp = torch.nan_to_num(detail_hyp, nan=0.0)
            detail_hyp = torch.clamp(detail_hyp, 0.0, 1.0)
            hypotheses.append(detail_hyp)
            
        if 'denoise' in self.active_decoders:
            denoise_hyp = self.denoise_decoder(encoder_features)
            if torch.isnan(denoise_hyp).any():
                denoise_hyp = torch.nan_to_num(denoise_hyp, nan=0.0)
            denoise_hyp = torch.clamp(denoise_hyp, 0.0, 1.0)
            hypotheses.append(denoise_hyp)
        
        # Try regular fusion first
        try:
            fused_image = self.fusion(hypotheses, degradation_maps)
            if torch.isnan(fused_image).any():
                raise ValueError("NaN detected in fusion output")
        except Exception as e:
            # Fall back to stable fusion if regular fusion fails
            print(f"Warning: Fusion failed ({e}), falling back to stable fusion")
            fused_image = self.stable_fusion(hypotheses)
            
        # Ensure fused image is valid
        if torch.isnan(fused_image).any():
            fused_image = torch.nan_to_num(fused_image, nan=0.0)
        fused_image = torch.clamp(fused_image, 0.0, 1.0)
        
        # Return outputs in consistent format
        return {
            'restored_image': fused_image,
            'degradation_maps': degradation_maps,
            'hypotheses': hypotheses
        }


class NoRefinementModel(nn.Module):
    """Model for no_refinement ablation"""
    def __init__(self, in_channels=3, base_channels=64, num_downs=5, 
                 num_degradation_types=4, norm_type='instance', activation='leaky_relu',
                 fusion_type='learned', **kwargs):
        super(NoRefinementModel, self).__init__()
        
        # Use standard DGMHRN model with enhanced stability
        self.model = DGMHRN(
            in_channels=in_channels,
            base_channels=base_channels,
            num_downs=num_downs,
            num_degradation_types=num_degradation_types,
            norm_type=norm_type,
            activation=activation,
            fusion_type=fusion_type
        )
        
        # Initialize with smaller weights
        self._initialize_weights()
        
        # Initialize dynamic layers
        self._init_dynamic_layers()
    
    def _initialize_weights(self):
        """Initialize weights to ensure stable training"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # Scale down weights to prevent explosion
                with torch.no_grad():
                    m.weight.data *= 0.3
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def _init_dynamic_layers(self):
        """Initialize any dynamic layers with a dummy forward pass"""
        if next(self.parameters()).is_cuda:
            device = next(self.parameters()).device
            dummy_input = torch.zeros(1, 3, 64, 64, device=device)
        else:
            dummy_input = torch.zeros(1, 3, 64, 64)
            
        # Run a dummy forward pass in eval mode to initialize all dynamic layers
        self.eval()
        with torch.no_grad():
            try:
                _ = self.forward(dummy_input)
            except Exception as e:
                print(f"Warning during dynamic layer initialization: {e}")
                # If 64x64 is too small, try a larger size
                try:
                    dummy_input = torch.zeros(1, 3, 128, 128, device=dummy_input.device)
                    _ = self.forward(dummy_input)
                except Exception as e:
                    print(f"Failed to initialize dynamic layers: {e}")
        self.train()  # Set back to training mode
    
    def forward(self, x):
        # Validate input
        if torch.isnan(x).any():
            x = torch.nan_to_num(x, nan=0.0)
            
        # Forward pass through base model
        outputs = self.model(x)
        
        # Ensure outputs are valid
        if torch.isnan(outputs['restored_image']).any():
            outputs['restored_image'] = torch.nan_to_num(outputs['restored_image'], nan=0.0)
            outputs['restored_image'] = torch.clamp(outputs['restored_image'], 0.0, 1.0)
            
        if torch.isnan(outputs['degradation_maps']).any():
            outputs['degradation_maps'] = torch.nan_to_num(outputs['degradation_maps'], nan=0.0)
            
        for i in range(len(outputs['hypotheses'])):
            if torch.isnan(outputs['hypotheses'][i]).any():
                outputs['hypotheses'][i] = torch.nan_to_num(outputs['hypotheses'][i], nan=0.0)
                outputs['hypotheses'][i] = torch.clamp(outputs['hypotheses'][i], 0.0, 1.0)
                
        return outputs


class WithRefinementModel(nn.Module):
    """Model for with_refinement ablation"""
    def __init__(self, in_channels=3, base_channels=64, num_downs=5, 
                 num_degradation_types=4, norm_type='instance', activation='leaky_relu',
                 fusion_type='learned', **kwargs):
        super(WithRefinementModel, self).__init__()
        
        # Create base model
        self.base_model = DGMHRN(
            in_channels=in_channels,
            base_channels=base_channels,
            num_downs=num_downs,
            num_degradation_types=num_degradation_types,
            norm_type=norm_type,
            activation=activation,
            fusion_type=fusion_type
        )
        
        # Create progressive model with enhanced stability
        self.progressive_model = TIDE(
            base_model=self.base_model,
            num_degradation_types=num_degradation_types,
            norm_type=norm_type,
            activation=activation
        )
        
        # Initialize with smaller weights
        self._initialize_weights()
        
        # Initialize dynamic layers
        self._init_dynamic_layers()
    
    def _initialize_weights(self):
        """Initialize weights to ensure stable training"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # Scale down weights to prevent explosion
                with torch.no_grad():
                    m.weight.data *= 0.3
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def _init_dynamic_layers(self):
        """Initialize any dynamic layers with a dummy forward pass"""
        if next(self.parameters()).is_cuda:
            device = next(self.parameters()).device
            dummy_input = torch.zeros(1, 3, 64, 64, device=device)
        else:
            dummy_input = torch.zeros(1, 3, 64, 64)
            
        # Run a dummy forward pass in eval mode to initialize all dynamic layers
        self.eval()
        with torch.no_grad():
            try:
                _ = self.forward(dummy_input)
            except Exception as e:
                print(f"Warning during dynamic layer initialization: {e}")
                # If 64x64 is too small, try a larger size
                try:
                    dummy_input = torch.zeros(1, 3, 128, 128, device=dummy_input.device)
                    _ = self.forward(dummy_input)
                except Exception as e:
                    print(f"Failed to initialize dynamic layers: {e}")
        self.train()  # Set back to training mode
    
    def forward(self, x):
        # Validate input
        if torch.isnan(x).any():
            x = torch.nan_to_num(x, nan=0.0)
            
        # Forward pass through progressive model
        outputs = self.progressive_model(x)
        
        # Ensure outputs are valid
        if torch.isnan(outputs['restored_image']).any():
            outputs['restored_image'] = torch.nan_to_num(outputs['restored_image'], nan=0.0)
            outputs['restored_image'] = torch.clamp(outputs['restored_image'], 0.0, 1.0)
            
        if 'initial_restoration' in outputs and torch.isnan(outputs['initial_restoration']).any():
            outputs['initial_restoration'] = torch.nan_to_num(outputs['initial_restoration'], nan=0.0)
            outputs['initial_restoration'] = torch.clamp(outputs['initial_restoration'], 0.0, 1.0)
            
        if 'refinement' in outputs and torch.isnan(outputs['refinement']).any():
            outputs['refinement'] = torch.nan_to_num(outputs['refinement'], nan=0.0)
            
        return outputs
    
    @property
    def enable_refinement(self):
        return self.progressive_model.enable_refinement
    
    @enable_refinement.setter
    def enable_refinement(self, value):
        self.progressive_model.enable_refinement = value


class RefinementMagnitudeModel(nn.Module):
    """Ablation model with configurable refinement magnitude scaling"""
    def __init__(self, in_channels=3, base_channels=64, num_downs=5,
                 num_degradation_types=4, norm_type='instance', activation='leaky_relu',
                 fusion_type='learned', refinement_scale=1.0, **kwargs):
        super(RefinementMagnitudeModel, self).__init__()
        
        # Create base model
        self.base_model = DGMHRN(
            in_channels=in_channels,
            base_channels=base_channels,
            num_downs=num_downs,
            num_degradation_types=num_degradation_types,
            norm_type=norm_type,
            activation=activation,
            fusion_type=fusion_type
        )
        
        # Create the progressive model
        self.progressive_model = TIDE(
            base_model=self.base_model,
            num_degradation_types=num_degradation_types,
            norm_type=norm_type,
            activation=activation
        )
        
        # Set the refinement scale
        self.refinement_scale = refinement_scale
        
        # Apply scale to all refinement components
        with torch.no_grad():
            # Scale the refinement fusion scale factor
            self.progressive_model.refinement_fusion.scale_factor.data *= refinement_scale
            
            # Scale each expert's scale factor
            for expert in self.progressive_model.refinement_experts:
                expert.scale_factor.data *= refinement_scale
        
        # Initialize with smaller weights
        self._initialize_weights()
        
        # Initialize dynamic layers
        self._init_dynamic_layers()
        
    def _initialize_weights(self):
        """Initialize weights for refinement components"""
        for m in self.progressive_model.residual_estimator.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # Scale down weights to prevent explosion
                with torch.no_grad():
                    m.weight.data *= 0.3
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def _init_dynamic_layers(self):
        """Initialize any dynamic layers with a dummy forward pass"""
        if next(self.parameters()).is_cuda:
            device = next(self.parameters()).device
            dummy_input = torch.zeros(1, 3, 64, 64, device=device)
        else:
            dummy_input = torch.zeros(1, 3, 64, 64)
            
        # Run a dummy forward pass in eval mode to initialize all dynamic layers
        self.eval()
        with torch.no_grad():
            try:
                _ = self.forward(dummy_input)
            except Exception as e:
                print(f"Warning during dynamic layer initialization: {e}")
                # If 64x64 is too small, try a larger size
                try:
                    dummy_input = torch.zeros(1, 3, 128, 128, device=dummy_input.device)
                    _ = self.forward(dummy_input)
                except Exception as e:
                    print(f"Failed to initialize dynamic layers: {e}")
        self.train()  # Set back to training mode
                
    def get_refinement_scale(self):
        """Return the current refinement scale"""
        return self.refinement_scale
    
    def set_refinement_scale(self, scale):
        """Set a new refinement scale dynamically"""
        # Calculate adjustment ratio
        adjustment = scale / self.refinement_scale
        
        with torch.no_grad():
            # Adjust fusion scale factor
            self.progressive_model.refinement_fusion.scale_factor.data *= adjustment
            
            # Adjust each expert's scale factor
            for expert in self.progressive_model.refinement_experts:
                expert.scale_factor.data *= adjustment
                
        # Update current scale
        self.refinement_scale = scale
        return self
    
    def forward(self, x):
        # Validate input
        if torch.isnan(x).any():
            x = torch.nan_to_num(x, nan=0.0)
            
        # Use the progressive model's forward pass
        outputs = self.progressive_model(x)
        
        # Ensure all outputs are valid
        for key in outputs:
            if isinstance(outputs[key], torch.Tensor):
                if torch.isnan(outputs[key]).any():
                    outputs[key] = torch.nan_to_num(outputs[key], nan=0.0)
                if key == 'restored_image' or key == 'initial_restoration':
                    outputs[key] = torch.clamp(outputs[key], 0.0, 1.0)
                    
        return outputs
    
    @property
    def enable_refinement(self):
        return self.progressive_model.enable_refinement
    
    @enable_refinement.setter
    def enable_refinement(self, value):
        self.progressive_model.enable_refinement = value


# Factory function to create the appropriate ablation model
def create_ablation_model(ablation_type, model_params):
    """
    Create an ablation model based on the specified type and parameters
    
    Args:
        ablation_type: Type of ablation to create
        model_params: Dictionary of model parameters
        
    Returns:
        Initialized ablation model
    """
    print(f"Creating ablation model of type: {ablation_type}")
    
    # Create a clean copy of model_params to avoid modifying the original
    params = model_params.copy()
    
    if ablation_type == 'no_degradation_maps':
        return NoDegradationMapsDGMHRN(**params)
    
    elif ablation_type == 'single_hypothesis':
        return SingleHypothesisDGMHRN(**params)
    
    elif ablation_type == 'fusion_type':
        # Requires 'fusion_type' in model_params
        assert 'fusion_type' in params, "fusion_type must be specified for fusion_type ablation"
        return FusionTypeDGMHRN(**params)
    
    elif ablation_type == 'decoder_types':
        # Requires 'decoder_types' in model_params
        assert 'decoder_types' in params, "decoder_types must be specified for decoder_types ablation"
        return DecoderCombinationDGMHRN(**params)
    
    elif ablation_type == 'no_diversity_loss':
        # Use standard DGMHRN, diversity loss will be disabled in loss function
        return DGMHRN(**params)
    
    elif ablation_type == 'no_refinement':
        variant = params.pop('variant', 'no_refinement')
        if variant == 'no_refinement':
            return NoRefinementModel(**params)
        else:  # with_refinement
            return WithRefinementModel(**params)
    
    elif ablation_type == 'refinement_magnitude':
        # Requires 'refinement_scale' in model_params
        assert 'refinement_scale' in params, "refinement_scale must be specified for refinement_magnitude ablation"
        return RefinementMagnitudeModel(**params)
    
    else:
        # Default case - use standard DGMHRN
        print(f"Warning: Unknown ablation type '{ablation_type}', using standard DGMHRN")
        return DGMHRN(**params)
    