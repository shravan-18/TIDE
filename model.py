import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Basic convolutional block with normalization and activation"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, 
                 use_norm=True, norm_type='instance', activation='leaky_relu', leaky_slope=0.2):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        
        self.use_norm = use_norm
        if use_norm:
            if norm_type == 'batch':
                self.norm = nn.BatchNorm2d(out_channels)
            elif norm_type == 'instance':
                self.norm = nn.InstanceNorm2d(out_channels)
            elif norm_type == 'none':
                self.use_norm = False
            else:
                raise ValueError(f"Unsupported normalization type: {norm_type}")
        
        self.activation = None
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(leaky_slope, inplace=True)
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation != 'none':
            raise ValueError(f"Unsupported activation: {activation}")
            
    def forward(self, x):
        x = self.conv(x)
        if self.use_norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class ResidualBlock(nn.Module):
    """Residual block for feature refinement"""
    def __init__(self, channels, norm_type='instance', activation='leaky_relu'):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvBlock(channels, channels, kernel_size=3, padding=1, 
                               norm_type=norm_type, activation=activation)
        self.conv2 = ConvBlock(channels, channels, kernel_size=3, padding=1, 
                               norm_type=norm_type, activation='none')
        
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            self.activation = nn.Identity()
            
    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + residual
        return self.activation(x)


class DownsampleBlock(nn.Module):
    """Downsampling block for encoder"""
    def __init__(self, in_channels, out_channels, norm_type='instance', activation='leaky_relu'):
        super(DownsampleBlock, self).__init__()
        self.conv = ConvBlock(in_channels, out_channels, kernel_size=4, stride=2, padding=1,
                              norm_type=norm_type, activation=activation)
    
    def forward(self, x):
        return self.conv(x)


class UpsampleBlock(nn.Module):
    """Upsampling block for decoder with skip connection support"""
    def __init__(self, in_channels, out_channels, skip_channels=0, norm_type='instance', activation='leaky_relu',
                 use_dropout=False, dropout_rate=0.5):
        super(UpsampleBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.skip_channels = skip_channels
        
        self.channel_calib = None
        if skip_channels > 0:
            self.adaptive_conv = True
        else:
            self.adaptive_conv = False
            self.conv = ConvBlock(in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                                  norm_type=norm_type, activation=activation)
        
        self.out_channels = out_channels
        self.norm_type = norm_type
        self.activation = activation
        
        self.use_dropout = use_dropout
        if use_dropout:
            self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x, skip=None):
        x = self.upsample(x)
        
        if skip is not None:
            if x.size() != skip.size():
                x = F.interpolate(x, size=skip.size()[2:], mode='bilinear', align_corners=True)
            x = torch.cat([x, skip], dim=1)
            
            actual_in_channels = x.size(1)
            if not hasattr(self, 'conv') or self.conv is None:
                self.conv = ConvBlock(actual_in_channels, self.out_channels, kernel_size=3, stride=1, padding=1,
                                      norm_type=self.norm_type, activation=self.activation).to(x.device)
            elif self.conv.conv.in_channels != actual_in_channels:
                self.conv = ConvBlock(actual_in_channels, self.out_channels, kernel_size=3, stride=1, padding=1,
                                      norm_type=self.norm_type, activation=self.activation).to(x.device)
        else:
            if not hasattr(self, 'conv') or self.conv is None:
                actual_in_channels = x.size(1)
                self.conv = ConvBlock(actual_in_channels, self.out_channels, kernel_size=3, stride=1, padding=1,
                                      norm_type=self.norm_type, activation=self.activation).to(x.device)
        
        x = self.conv(x)
        if self.use_dropout:
            x = self.dropout(x)
        return x


class Encoder(nn.Module):
    """Encoder E: Extracts hierarchical features F from input image I"""
    def __init__(self, in_channels=3, base_channels=64, num_downs=5, norm_type='instance',
                 activation='leaky_relu'):
        super(Encoder, self).__init__()
        
        self.init_conv = ConvBlock(in_channels, base_channels, kernel_size=7, padding=3,
                                  norm_type=norm_type, activation=activation)
        
        self.down_blocks = nn.ModuleList()
        self.features = [base_channels]
        in_ch = base_channels
        
        for i in range(num_downs):
            out_ch = min(in_ch * 2, 512)
            self.down_blocks.append(DownsampleBlock(in_ch, out_ch, norm_type, activation))
            self.features.append(out_ch)
            in_ch = out_ch
            
        self.res_blocks = nn.Sequential(
            ResidualBlock(in_ch, norm_type, activation),
            ResidualBlock(in_ch, norm_type, activation)
        )
        
    def forward(self, x):
        features = [self.init_conv(x)]
        
        for block in self.down_blocks:
            features.append(block(features[-1]))
            
        features[-1] = self.res_blocks(features[-1])
        
        return features


class DegradationEstimator(nn.Module):
    """Estimates degradation maps M for K degradation types (color, contrast, detail, noise)"""
    def __init__(self, in_channels=3, base_channels=32, num_degradation_types=4,
                 norm_type='instance', activation='leaky_relu'):
        super(DegradationEstimator, self).__init__()
        
        self.init_conv = ConvBlock(in_channels, base_channels, kernel_size=7, padding=3,
                                  norm_type=norm_type, activation=activation)
        
        self.down1 = DownsampleBlock(base_channels, base_channels*2, norm_type, activation)
        self.down2 = DownsampleBlock(base_channels*2, base_channels*4, norm_type, activation)
        self.down3 = DownsampleBlock(base_channels*4, base_channels*8, norm_type, activation)
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(base_channels*8, base_channels*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(base_channels*4, num_degradation_types)
        )
        
        self.up_blocks = nn.ModuleList([
            UpsampleBlock(base_channels*8, base_channels*4, norm_type=norm_type, activation=activation),
            UpsampleBlock(base_channels*4, base_channels*2, norm_type=norm_type, activation=activation),
            UpsampleBlock(base_channels*2, base_channels, norm_type=norm_type, activation=activation)
        ])
        
        self.final_conv = nn.Sequential(
            ConvBlock(base_channels, base_channels, kernel_size=3, padding=1,
                     norm_type=norm_type, activation=activation),
            nn.Conv2d(base_channels, num_degradation_types, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.init_conv(x)
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        
        global_context = self.global_pool(x3)
        global_context = global_context.view(global_context.size(0), -1)
        global_weights = self.fc(global_context)
        global_weights = torch.sigmoid(global_weights).unsqueeze(-1).unsqueeze(-1)
        
        x = x3
        for up_block in self.up_blocks:
            x = up_block(x)
        
        degradation_maps = self.final_conv(x)
        degradation_maps = degradation_maps * global_weights
        
        return degradation_maps


class SpecializedDecoder(nn.Module):
    """Base class for specialized restoration decoders Fk"""
    def __init__(self, encoder_features, norm_type='instance', activation='leaky_relu',
                 use_skip_connections=True):
        super(SpecializedDecoder, self).__init__()
        
        self.encoder_features = encoder_features
        self.use_skip_connections = use_skip_connections
        
        self.up_blocks = nn.ModuleList()
        num_levels = len(encoder_features) - 1
        
        for i in range(num_levels):
            level = num_levels - i - 1
            in_channels = encoder_features[level + 1]
            out_channels = encoder_features[level]
            skip_channels = out_channels if use_skip_connections and level > 0 else 0
            
            self.up_blocks.append(
                UpsampleBlock(in_channels, out_channels, skip_channels,
                            norm_type=norm_type, activation=activation)
            )
        
        self.final_conv = nn.Sequential(
            ConvBlock(encoder_features[0], 32, kernel_size=3, padding=1,
                     norm_type=norm_type, activation=activation),
            nn.Conv2d(32, 3, kernel_size=7, padding=3),
            nn.Tanh()
        )
        
    def forward(self, encoder_features):
        x = encoder_features[-1]
        
        for i, up_block in enumerate(self.up_blocks):
            level = len(encoder_features) - i - 2
            if self.use_skip_connections and level > 0:
                skip = encoder_features[level]
                x = up_block(x, skip)
            else:
                x = up_block(x)
        
        hypothesis = self.final_conv(x)
        hypothesis = (hypothesis + 1) / 2
        
        return hypothesis


class ColorRestorationDecoder(SpecializedDecoder):
    """Color restoration decoder F1: Corrects wavelength-dependent attenuation"""
    def __init__(self, encoder_features, norm_type='instance', activation='leaky_relu'):
        super(ColorRestorationDecoder, self).__init__(encoder_features, norm_type, activation)
        
        self.channel_attention = nn.ModuleList()
        for feat_dim in reversed(encoder_features[1:]):
            self.channel_attention.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Conv2d(feat_dim, feat_dim // 4, 1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(feat_dim // 4, feat_dim, 1),
                    nn.Sigmoid()
                )
            )
        
    def forward(self, encoder_features):
        x = encoder_features[-1]
        
        for i, (up_block, attn) in enumerate(zip(self.up_blocks, self.channel_attention)):
            x = x * attn(x)
            
            level = len(encoder_features) - i - 2
            if self.use_skip_connections and level > 0:
                skip = encoder_features[level]
                x = up_block(x, skip)
            else:
                x = up_block(x)
        
        hypothesis = self.final_conv(x)
        hypothesis = (hypothesis + 1) / 2
        
        return hypothesis


class ContrastEnhancementDecoder(SpecializedDecoder):
    """Contrast enhancement decoder F2: Addresses scattering-induced contrast loss"""
    def __init__(self, encoder_features, norm_type='instance', activation='leaky_relu'):
        super(ContrastEnhancementDecoder, self).__init__(encoder_features, norm_type, activation)
        
        self.contrast_blocks = nn.ModuleList()
        for feat_dim in reversed(encoder_features[1:]):
            self.contrast_blocks.append(
                nn.Sequential(
                    ResidualBlock(feat_dim, norm_type, activation),
                    ResidualBlock(feat_dim, norm_type, activation)
                )
            )
        
    def forward(self, encoder_features):
        x = encoder_features[-1]
        
        for i, (up_block, contrast_block) in enumerate(zip(self.up_blocks, self.contrast_blocks)):
            x = contrast_block(x)
            
            level = len(encoder_features) - i - 2
            if self.use_skip_connections and level > 0:
                skip = encoder_features[level]
                x = up_block(x, skip)
            else:
                x = up_block(x)
        
        hypothesis = self.final_conv(x)
        hypothesis = (hypothesis + 1) / 2
        
        return hypothesis


class DetailRecoveryDecoder(SpecializedDecoder):
    """Detail recovery decoder F3: Recovers fine structures via cascaded enhancement"""
    def __init__(self, encoder_features, norm_type='instance', activation='leaky_relu'):
        super(DetailRecoveryDecoder, self).__init__(encoder_features, norm_type, activation)
        
        self.detail_blocks = nn.ModuleList()
        for feat_dim in reversed(encoder_features[1:]):
            self.detail_blocks.append(
                nn.Sequential(
                    ResidualBlock(feat_dim, norm_type, activation),
                    ResidualBlock(feat_dim, norm_type, activation),
                    ResidualBlock(feat_dim, norm_type, activation)
                )
            )
        
    def forward(self, encoder_features):
        x = encoder_features[-1]
        
        for i, (up_block, detail_block) in enumerate(zip(self.up_blocks, self.detail_blocks)):
            x_cumulative = x
            for res_block in detail_block:
                x_res = res_block(x_cumulative)
                x_cumulative = x_cumulative + x_res
            x = x_cumulative
            
            level = len(encoder_features) - i - 2
            if self.use_skip_connections and level > 0:
                skip = encoder_features[level]
                x = up_block(x, skip)
            else:
                x = up_block(x)
        
        hypothesis = self.final_conv(x)
        hypothesis = (hypothesis + 1) / 2
        
        return hypothesis


class DenoisingDecoder(SpecializedDecoder):
    """Denoising decoder F4: Removes noise from suspended particles using group convolutions"""
    def __init__(self, encoder_features, norm_type='instance', activation='leaky_relu'):
        super(DenoisingDecoder, self).__init__(encoder_features, norm_type, activation)
        
        self.denoise_blocks = nn.ModuleList()
        for feat_dim in reversed(encoder_features[1:]):
            groups = min(max(feat_dim // 8, 1), feat_dim)
            self.denoise_blocks.append(
                nn.Sequential(
                    nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=1, groups=groups),
                    nn.InstanceNorm2d(feat_dim) if norm_type == 'instance' else nn.BatchNorm2d(feat_dim),
                    nn.LeakyReLU(0.2, inplace=True),
                    ResidualBlock(feat_dim, norm_type, activation)
                )
            )
        
    def forward(self, encoder_features):
        x = encoder_features[-1]
        
        for i, (up_block, denoise_block) in enumerate(zip(self.up_blocks, self.denoise_blocks)):
            x = denoise_block(x)
            
            level = len(encoder_features) - i - 2
            if self.use_skip_connections and level > 0:
                skip = encoder_features[level]
                x = up_block(x, skip)
            else:
                x = up_block(x)
        
        hypothesis = self.final_conv(x)
        hypothesis = (hypothesis + 1) / 2
        
        return hypothesis


class AdaptiveFusion(nn.Module):
    """Adaptive fusion mechanism: Maps degradation maps M to fusion weights W"""
    def __init__(self, num_hypotheses, num_degradation_types, norm_type='instance',
                 activation='leaky_relu', fusion_strategy='learned'):
        super(AdaptiveFusion, self).__init__()
        
        self.fusion_strategy = fusion_strategy
        self.num_hypotheses = num_hypotheses
        
        if fusion_strategy == 'learned':
            self.weight_generator = nn.Sequential(
                ConvBlock(num_degradation_types, 32, kernel_size=3, padding=1,
                         norm_type=norm_type, activation=activation),
                ResidualBlock(32, norm_type, activation),
                nn.Conv2d(32, num_hypotheses, kernel_size=1),
                nn.Softmax(dim=1)
            )
        
        self.temperature = nn.Parameter(torch.ones(1))
        
    def forward(self, hypotheses, degradation_maps):
        hypotheses_stack = torch.stack(hypotheses, dim=1)
        
        if self.fusion_strategy == 'learned':
            fusion_weights = self.weight_generator(degradation_maps)
        elif self.fusion_strategy == 'direct':
            fusion_weights = degradation_maps[:, :self.num_hypotheses]
            fusion_weights = F.softmax(fusion_weights / self.temperature, dim=1)
        elif self.fusion_strategy == 'average':
            B, _, H, W = degradation_maps.shape
            fusion_weights = torch.ones(B, self.num_hypotheses, H, W, device=degradation_maps.device)
            fusion_weights = fusion_weights / self.num_hypotheses
        else:
            raise ValueError(f"Unknown fusion strategy: {self.fusion_strategy}")
        
        fusion_weights = fusion_weights.unsqueeze(2)
        weighted_hypotheses = hypotheses_stack * fusion_weights
        initial_restoration = weighted_hypotheses.sum(dim=1)
        
        return initial_restoration, fusion_weights.squeeze(2)


class DGMHRN(nn.Module):
    """Base restoration model: Generates initial restoration J_hat_1 J_hat_1"""
    def __init__(self, in_channels=3, base_channels=64, num_downs=5, 
                 num_degradation_types=4, norm_type='instance', activation='leaky_relu',
                 use_skip_connections=True, fusion_strategy='learned'):
        super(DGMHRN, self).__init__()
        
        self.encoder = Encoder(in_channels, base_channels, num_downs, norm_type, activation)
        
        self.degradation_estimator = DegradationEstimator(
            in_channels, base_channels=32, num_degradation_types=num_degradation_types,
            norm_type=norm_type, activation=activation
        )
        
        encoder_features = self.encoder.features
        
        self.color_decoder = ColorRestorationDecoder(encoder_features, norm_type, activation)
        self.contrast_decoder = ContrastEnhancementDecoder(encoder_features, norm_type, activation)
        self.detail_decoder = DetailRecoveryDecoder(encoder_features, norm_type, activation)
        self.denoise_decoder = DenoisingDecoder(encoder_features, norm_type, activation)
        
        self.fusion = AdaptiveFusion(
            num_hypotheses=4,
            num_degradation_types=num_degradation_types,
            norm_type=norm_type,
            activation=activation,
            fusion_strategy=fusion_strategy
        )
        
    def forward(self, x):
        F = self.encoder(x)
        M = self.degradation_estimator(x)
        
        H1 = self.color_decoder(F)
        H2 = self.contrast_decoder(F)
        H3 = self.detail_decoder(F)
        H4 = self.denoise_decoder(F)
        
        hypotheses = [H1, H2, H3, H4]
        
        J_hat_1, fusion_weights = self.fusion(hypotheses, M)
        J_hat_1 = torch.clamp(J_hat_1, 0, 1)
        
        return {
            'restored_image': J_hat_1,
            'degradation_maps': M,
            'hypotheses': hypotheses,
            'fusion_weights': fusion_weights
        }


class ResidualDegradationEstimator(nn.Module):
    """Estimates residual degradation maps Mr maps Mr via differential analysis"""
    def __init__(self, in_channels=6, base_channels=32, num_degradation_types=4,
                 norm_type='instance', activation='leaky_relu'):
        super(ResidualDegradationEstimator, self).__init__()
        
        self.difference_encoder = nn.Sequential(
            ConvBlock(3, base_channels, kernel_size=3, padding=1,
                     norm_type=norm_type, activation=activation),
            ConvBlock(base_channels, base_channels, kernel_size=3, padding=1,
                     norm_type=norm_type, activation=activation)
        )
        
        self.residual_encoder = nn.Sequential(
            ConvBlock(in_channels, base_channels*2, kernel_size=7, padding=3,
                     norm_type=norm_type, activation=activation),
            DownsampleBlock(base_channels*2, base_channels*4, norm_type, activation),
            ResidualBlock(base_channels*4, norm_type, activation),
            ResidualBlock(base_channels*4, norm_type, activation)
        )
        
        self.upsample = UpsampleBlock(base_channels*4, base_channels*2,
                                     norm_type=norm_type, activation=activation)
        
        self.fusion_conv = ConvBlock(base_channels*3, base_channels*2, kernel_size=3, padding=1,
                                     norm_type=norm_type, activation=activation)
        
        self.output_conv = nn.Sequential(
            ConvBlock(base_channels*2, base_channels, kernel_size=3, padding=1,
                     norm_type=norm_type, activation=activation),
            nn.Conv2d(base_channels, num_degradation_types, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.alpha = nn.Parameter(torch.tensor(1.0))
        
    def forward(self, original_img, initial_restoration):
        D = torch.abs(original_img - initial_restoration)
        
        diff_features = self.difference_encoder(D)
        
        concat_input = torch.cat([original_img, initial_restoration], dim=1)
        residual_features = self.residual_encoder(concat_input)
        
        residual_features = self.upsample(residual_features)
        
        if residual_features.size()[2:] != diff_features.size()[2:]:
            residual_features = F.interpolate(residual_features, size=diff_features.size()[2:],
                                            mode='bilinear', align_corners=True)
        
        combined = torch.cat([residual_features, diff_features], dim=1)
        combined = self.fusion_conv(combined)
        
        Mr = self.output_conv(combined)
        Mr = Mr * (1 + self.alpha * diff_features.mean(dim=1, keepdim=True))
        
        return Mr


class RefinementExpert(nn.Module):
    """Refinement expert Ek: Generates correction term Ck for specific degradation type"""
    def __init__(self, in_channels=6, base_channels=32, expert_type='color',
                 norm_type='instance', activation='leaky_relu'):
        super(RefinementExpert, self).__init__()
        
        self.expert_type = expert_type
        
        self.init_conv = ConvBlock(in_channels, base_channels, kernel_size=7, padding=3,
                                  norm_type=norm_type, activation=activation)
        
        self.res_block1 = ResidualBlock(base_channels, norm_type, activation)
        self.res_block2 = ResidualBlock(base_channels, norm_type, activation)
        
        if expert_type == 'color':
            self.specific_block = nn.Sequential(
                ConvBlock(base_channels, base_channels*2, kernel_size=3, padding=1,
                        norm_type=norm_type, activation=activation),
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(base_channels*2, base_channels*2, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(base_channels*2, base_channels, kernel_size=1),
                nn.Upsample(scale_factor=256, mode='bilinear', align_corners=True),
                nn.Conv2d(base_channels, 3, kernel_size=1),
                nn.Tanh()
            )
        elif expert_type == 'contrast':
            self.specific_block = nn.Sequential(
                ResidualBlock(base_channels, norm_type, activation),
                ResidualBlock(base_channels, norm_type, activation),
                ConvBlock(base_channels, base_channels//2, kernel_size=3, padding=1,
                        norm_type=norm_type, activation=activation),
                nn.Conv2d(base_channels//2, 3, kernel_size=1),
                nn.Tanh()
            )
        elif expert_type == 'detail':
            self.specific_block = nn.Sequential(
                ConvBlock(base_channels, base_channels*2, kernel_size=3, padding=1,
                        norm_type=norm_type, activation=activation),
                ResidualBlock(base_channels*2, norm_type, activation),
                ConvBlock(base_channels*2, base_channels, kernel_size=3, padding=1,
                        norm_type=norm_type, activation=activation),
                ConvBlock(base_channels, base_channels//2, kernel_size=3, padding=1,
                        norm_type=norm_type, activation=activation),
                nn.Conv2d(base_channels//2, 3, kernel_size=1),
                nn.Tanh()
            )
        elif expert_type == 'noise':
            groups = base_channels // 4 if base_channels >= 4 else 1
            self.specific_block = nn.Sequential(
                ConvBlock(base_channels, base_channels, kernel_size=3, padding=1, 
                        norm_type=norm_type, activation=activation),
                nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1, groups=groups),
                nn.LeakyReLU(0.2, inplace=True),
                ConvBlock(base_channels, base_channels//2, kernel_size=3, padding=1,
                        norm_type=norm_type, activation=activation),
                nn.Conv2d(base_channels//2, 3, kernel_size=1),
                nn.Tanh()
            )
        
        self.scale_factor = nn.Parameter(torch.tensor(0.1))
        
    def forward(self, original_img, initial_restoration):
        x = torch.cat([original_img, initial_restoration], dim=1)
        
        x = self.init_conv(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        
        correction = self.specific_block(x)
        correction = correction * torch.sigmoid(self.scale_factor)
        
        return correction


class SafetyGatedFusion(nn.Module):
    """Safety-gated fusion: Combines correction terms with adaptive gating"""
    def __init__(self, num_refinements, num_degradation_types, norm_type='instance',
                 activation='leaky_relu'):
        super(SafetyGatedFusion, self).__init__()
        
        self.mapping = nn.Sequential(
            ConvBlock(num_degradation_types, 32, kernel_size=3, padding=1,
                     norm_type=norm_type, activation=activation),
            ResidualBlock(32, norm_type, activation),
            nn.Conv2d(32, num_refinements, kernel_size=1),
            nn.Softmax(dim=1)
        )
        
        self.scale_factor = nn.Parameter(torch.tensor(0.5))
        
        self.safety_gate = nn.Sequential(
            ConvBlock(3, 16, kernel_size=3, padding=1, norm_type='none', activation=activation),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, refinements, residual_maps, initial_restoration=None):
        refinements_stack = torch.stack(refinements, dim=1)
        
        weights = self.mapping(residual_maps)
        weights = weights.unsqueeze(2)
        
        weighted_refinements = refinements_stack * weights
        C = weighted_refinements.sum(dim=1)
        
        C = C * torch.sigmoid(self.scale_factor)
        
        if initial_restoration is not None:
            G = self.safety_gate(initial_restoration)
            C = C * G
        
        return C


class TIDE(nn.Module):
    """TIDE: Two-stage inverse degradation estimation framework"""
    def __init__(self, base_model, num_degradation_types=4, norm_type='instance', 
                 activation='leaky_relu'):
        super(TIDE, self).__init__()
        
        self.base_model = base_model
        
        self.residual_estimator = ResidualDegradationEstimator(
            in_channels=6,
            base_channels=32,
            num_degradation_types=num_degradation_types,
            norm_type=norm_type,
            activation=activation
        )
        
        self.refinement_experts = nn.ModuleList([
            RefinementExpert(in_channels=6, base_channels=32, expert_type='color',
                           norm_type=norm_type, activation=activation),
            RefinementExpert(in_channels=6, base_channels=32, expert_type='contrast',
                           norm_type=norm_type, activation=activation),
            RefinementExpert(in_channels=6, base_channels=32, expert_type='detail',
                           norm_type=norm_type, activation=activation),
            RefinementExpert(in_channels=6, base_channels=32, expert_type='noise',
                           norm_type=norm_type, activation=activation)
        ])
        
        self.refinement_fusion = SafetyGatedFusion(
            num_refinements=len(self.refinement_experts),
            num_degradation_types=num_degradation_types,
            norm_type=norm_type,
            activation=activation
        )
        
        self.enable_refinement = True
        
    def forward(self, x):
        base_outputs = self.base_model(x)
        J_hat_1 = base_outputs['restored_image']
        
        if not self.enable_refinement:
            return base_outputs
        
        Mr = self.residual_estimator(x, J_hat_1)
        
        corrections = []
        for expert in self.refinement_experts:
            Ck = expert(x, J_hat_1)
            corrections.append(Ck)
        
        C = self.refinement_fusion(corrections, Mr, J_hat_1)
        
        J_hat = J_hat_1 + C
        J_hat = torch.clamp(J_hat, 0, 1)
        
        return {
            'restored_image': J_hat,
            'initial_restoration': J_hat_1,
            'refinement': C,
            'degradation_maps': base_outputs['degradation_maps'],
            'residual_maps': Mr,
            'hypotheses': base_outputs['hypotheses'],
            'corrections': corrections
        }
        
    def set_refinement_enabled(self, enabled=True):
        """Enable or disable refinement stage"""
        self.enable_refinement = enabled
        return self
