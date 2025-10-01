# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# Global channel configuration for Lite-style networks
LITE_CHANNELS = [32, 32, 64, 96, 128, 192, 256]

class SobelGradientLayer(nn.Module):
    """Fixed Sobel operator for spatial gradient computation."""
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]], dtype=torch.float32).unsqueeze(0) / 8.0
        sobel_y = torch.tensor([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]], dtype=torch.float32).unsqueeze(0) / 8.0
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] input feature map
        Returns:
            Ix, Iy: spatial gradients, each [B, 1, H, W]
        """
        if x.shape[1] == 1:
            Ix = F.conv2d(x, self.sobel_x, padding=1)
            Iy = F.conv2d(x, self.sobel_y, padding=1)
        else:
            x_mean = torch.mean(x, dim=1, keepdim=True)
            Ix = F.conv2d(x_mean, self.sobel_x, padding=1)
            Iy = F.conv2d(x_mean, self.sobel_y, padding=1)
        return Ix, Iy

def make_warp_grid(flow, inverse=True):
    """Generate a normalized grid for warping with optical flow."""
    B, C, H, W = flow.size()
    xx = torch.linspace(-1.0, 1.0, W, device=flow.device).view(1, 1, 1, W).expand(B, -1, H, -1)
    yy = torch.linspace(-1.0, 1.0, H, device=flow.device).view(1, 1, H, 1).expand(B, -1, -1, W)
    base_grid = torch.cat((xx, yy), 1)
    norm_flow = torch.stack([
        flow[:, 0] / ((W - 1.0) / 2.0),
        flow[:, 1] / ((H - 1.0) / 2.0)
    ], dim=1)
    if inverse:
        norm_flow = -norm_flow
    grid = (base_grid + norm_flow).permute(0, 2, 3, 1)
    return grid

class DownsampleBlock(nn.Module):
    """A block of three 3x3 convs, with optional downsampling on the first conv."""
    def __init__(self, in_channels, out_channels, downsample=True):
        super().__init__()
        stride = 2 if downsample else 1
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.LeakyReLU(0.1, inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.LeakyReLU(0.1, inplace=True)
        self.conv3 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu3 = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        return x

class Encoder(nn.Module):
    """Encoder for the gray branch, outputs multi-scale features."""
    def __init__(self, in_channels=2, num_layers=7):
        super().__init__()
        self.num_layers = num_layers
        self.sobel_layer = SobelGradientLayer()
        self.conv0 = DownsampleBlock(in_channels, LITE_CHANNELS[0], downsample=False)
        self.conv_layers = nn.ModuleList([
            DownsampleBlock(LITE_CHANNELS[i-1], LITE_CHANNELS[i])
            for i in range(1, num_layers)
        ])

    def forward(self, x, binary_x=None):
        """
        Args:
            x: [B, 2, H, W] gray image pair
            binary_x: [B, 2, H, W] (optional) binary image pair
        Returns:
            features: list of encoder outputs
            original_frames_pyramid: list of downsampled input images
            binary_frames_pyramid: list of downsampled binary images (if provided)
        """
        original_frames_pyramid = [x]
        binary_frames_pyramid = [binary_x] if binary_x is not None else None
        features = [self.conv0(x)]
        out = features[0]
        for i, layer in enumerate(self.conv_layers):
            downsampled = F.avg_pool2d(original_frames_pyramid[-1], 2, 2)
            original_frames_pyramid.append(downsampled)
            if binary_frames_pyramid:
                binary_frames_pyramid.append(F.avg_pool2d(binary_frames_pyramid[-1], 2, 2))
            out = layer(out)
            features.append(out)
        return features, original_frames_pyramid, binary_frames_pyramid

class FusionLayer(nn.Module):
    """Main fusion conv layer in decoder."""
    def __init__(self, fusion_in_ch, out_ch):
        super().__init__()
        self.main_conv = nn.Sequential(
            nn.Conv2d(fusion_in_ch, out_ch, 3, 1, 1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True)
        )
    def forward(self, x):
        return self.main_conv(x)

class FlowEstimator(nn.Module):
    """Flow estimator with 5 conv layers and a final prediction layer."""
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 128, 3, 1, 1)
        self.leaky_relu1 = nn.LeakyReLU(0.1, inplace=True)
        self.conv2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.leaky_relu2 = nn.LeakyReLU(0.1, inplace=True)
        self.conv3 = nn.Conv2d(128, 96, 3, 1, 1)
        self.leaky_relu3 = nn.LeakyReLU(0.1, inplace=True)
        self.conv4 = nn.Conv2d(96, 64, 3, 1, 1)
        self.leaky_relu4 = nn.LeakyReLU(0.1, inplace=True)
        self.conv5 = nn.Conv2d(64, 32, 3, 1, 1)
        self.leaky_relu5 = nn.LeakyReLU(0.1, inplace=True)
        self.predict_flow = nn.Conv2d(32, 2, 3, 1, 1)

    def forward(self, x):
        x = self.leaky_relu1(self.conv1(x))
        x = self.leaky_relu2(self.conv2(x))
        x = self.leaky_relu3(self.conv3(x))
        x = self.leaky_relu4(self.conv4(x))
        x = self.leaky_relu5(self.conv5(x))
        return self.predict_flow(x)

class Decoder(nn.Module):
    """Decoder for the gray branch, predicts multi-scale flow fields."""
    def __init__(self, params, num_layers=7):
        super().__init__()
        self.params = params
        self.num_layers = num_layers
        self.fusion_layers = nn.ModuleList()
        self.prediction_layers = nn.ModuleList()
        self.upconv_layers = nn.ModuleList()
        self.sobel_layer = SobelGradientLayer()
        self.confidence_pool_kernel = nn.AvgPool2d(5, 1, 2)
        
        for i in range(num_layers-1, -1, -1):
            in_ch_x = LITE_CHANNELS[i]
            out_ch_upconv = LITE_CHANNELS[i-1] if i > 0 else LITE_CHANNELS[-1]
            if i == num_layers-1:
                # x, frame1, frame2, frame1_warped, flow: in_ch_x + 1 + 1 + 1 + 2 = in_ch_x + 5
                fusion_in_ch = in_ch_x + 1 + 1 + 1 + 2
            else:
                # x, frame1_warped, frame2, diff, mul, flow_up, skip: in_ch_x + 1 + 1 + 1 + 1 + 2 + in_ch_x = 2*in_ch_x + 6
                fusion_in_ch = in_ch_x + 1 + 1 + 1 + 1 + 2 + in_ch_x
            self.fusion_layers.append(FusionLayer(fusion_in_ch, in_ch_x))
            self.prediction_layers.append(FlowEstimator(in_ch_x))
            if i > 0:
                self.upconv_layers.append(nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                    nn.Conv2d(in_ch_x, out_ch_upconv, 3, 1, 1),
                    nn.BatchNorm2d(out_ch_upconv),
                    nn.LeakyReLU(0.1, inplace=True)
                ))
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, encoder_features, original_frames_pyramid, initial_binary_flow_high_res=None, return_intermediate=False, binary_frames_pyramid=None):
        predictions, residuals, gradients, confidences = {}, {}, {}, {}
        x = encoder_features[-1]
        accumulated_flow = None
        guidance_dropout_prob = getattr(self.params.training, 'guidance_dropout_prob', 0.0) if hasattr(self, 'params') else 0.0
        intermediate_outputs = {'flows': [], 'residuals': [], 'smooth_errors': []} if return_intermediate else None
        
        for idx in range(self.num_layers):
            i = self.num_layers - 1 - idx
            frame1 = original_frames_pyramid[i][:, 0:1]
            frame2 = original_frames_pyramid[i][:, 1:2]
            current_level_base_flow = None
            frame1_warped = frame1
            
            if idx == 0:
                local_initial_guidance = initial_binary_flow_high_res
                if self.training and guidance_dropout_prob > 0 and torch.rand(1).item() < guidance_dropout_prob:
                    local_initial_guidance = None
                if local_initial_guidance is not None:
                    target_H, target_W = x.shape[2], x.shape[3]
                    H_hr, W_hr = local_initial_guidance.shape[2], local_initial_guidance.shape[3]
                    scale_val_w = float(target_W) / W_hr
                    scale_val_h = float(target_H) / H_hr
                    flow_u_scaled = local_initial_guidance[:, 0:1, :, :] * scale_val_w
                    flow_v_scaled = local_initial_guidance[:, 1:2, :, :] * scale_val_h
                    flow_hr_scaled = torch.cat([flow_u_scaled, flow_v_scaled], dim=1)
                    current_level_base_flow = F.adaptive_avg_pool2d(flow_hr_scaled, (target_H, target_W))
                    grid = make_warp_grid(current_level_base_flow, inverse=False)
                    frame1_warped = F.grid_sample(frame1, grid, align_corners=True, padding_mode='border')
            else:
                if accumulated_flow is not None:
                    scale_factor = frame1.shape[2] / accumulated_flow.shape[2]
                    flow_up = F.interpolate(accumulated_flow, size=frame1.shape[2:], mode='bilinear', align_corners=True) * scale_factor
                    current_level_base_flow = flow_up
                    grid = make_warp_grid(current_level_base_flow, inverse=False)
                    frame1_warped = F.grid_sample(frame1, grid, align_corners=True, padding_mode='border')
            
            # Prepare fusion inputs
            if idx == 0:
                fusion_inputs_list = [x, frame1, frame2, frame1_warped]
                if current_level_base_flow is not None:
                    fusion_inputs_list.append(current_level_base_flow)
                else:
                    B, _, H, W = x.shape
                    zeros_for_flow = torch.zeros(B, 2, H, W, device=x.device, dtype=x.dtype)
                    fusion_inputs_list.append(zeros_for_flow)
            else:
                diff_term = frame2 - frame1_warped
                mul_term = frame2 * frame1_warped
                fusion_inputs_list = [x, frame1_warped, frame2, diff_term, mul_term, current_level_base_flow, encoder_features[i]]
            
            # Fusion and prediction
            fused = torch.cat(fusion_inputs_list, dim=1)
            x_fused = self.fusion_layers[idx](fused)
            original_residual = self.prediction_layers[idx](x_fused)
            residuals[f'flow_level_{i}'] = original_residual
            
            # Compute gradients and confidence
            gray_frame1_level = original_frames_pyramid[i][:, 0:1, :, :]
            Ix, Iy = self.sobel_layer(gray_frame1_level)
            # Gradient clipping for numerical stability
            Ix = torch.clamp(Ix, -0.5, 0.5)
            Iy = torch.clamp(Iy, -0.5, 0.5)
            grad_mag = torch.sqrt(torch.clamp(Ix ** 2 + Iy ** 2, min=1e-8))
            alpha = 1.0
            raw_conf_edges = 1.0 - torch.exp(-alpha * grad_mag)
            min_e = torch.min(raw_conf_edges.view(raw_conf_edges.shape[0], -1), dim=1, keepdim=True)[0].view(raw_conf_edges.shape[0], 1, 1, 1)
            max_e = torch.max(raw_conf_edges.view(raw_conf_edges.shape[0], -1), dim=1, keepdim=True)[0].view(raw_conf_edges.shape[0], 1, 1, 1)
            norm_conf_edges = (raw_conf_edges - min_e) / (max_e - min_e + 1e-6)
            
            # Compute density-based confidence
            if binary_frames_pyramid is not None:
                binary_frame1_level = binary_frames_pyramid[i][:, 0:1, :, :]
                raw_conf_density = self.confidence_pool_kernel(binary_frame1_level)
                raw_conf_density = torch.clamp(raw_conf_density * 5.0, 0.0, 1.0)
                min_d = torch.min(raw_conf_density.view(raw_conf_density.shape[0], -1), dim=1, keepdim=True)[0].view(raw_conf_density.shape[0], 1, 1, 1)
                max_d = torch.max(raw_conf_density.view(raw_conf_density.shape[0], -1), dim=1, keepdim=True)[0].view(raw_conf_density.shape[0], 1, 1, 1)
                norm_conf_density = (raw_conf_density - min_d) / (max_d - min_d + 1e-6)
            else:
                norm_conf_density = torch.zeros_like(norm_conf_edges)
            
            # Combine edge and density confidence
            final_confidence = 0.2 * norm_conf_edges + 0.8 * norm_conf_density
            confidences[f'level_{i}'] = final_confidence
            
            # Update accumulated flow
            if current_level_base_flow is not None:
                accumulated_flow = current_level_base_flow + original_residual
            else:
                accumulated_flow = original_residual
            predictions[f'flow_level_{i}'] = accumulated_flow
            
            # Compute temporal gradient
            Ix, Iy = self.sobel_layer(frame1_warped)
            grid_for_It = make_warp_grid(original_residual, inverse=False)
            frame1_further_warped_by_residual = F.grid_sample(frame1_warped, grid_for_It, align_corners=True, padding_mode='border')
            It = frame1_further_warped_by_residual - frame1_warped
            gradients[f'level_{i}'] = {'Ix': Ix, 'Iy': Iy, 'It': It}
            
            # Upsample for next level
            if idx < self.num_layers - 1:
                x = self.upconv_layers[idx](x_fused)
            
            if return_intermediate:
                intermediate_outputs['flows'].append(accumulated_flow)
                intermediate_outputs['residuals'].append(original_residual)
                intermediate_outputs['smooth_errors'].append(torch.zeros_like(original_residual))
        
        predictions['final_flow'] = predictions['flow_level_0']
        if return_intermediate:
            return predictions, residuals, gradients, confidences, predictions['final_flow'], residuals, original_frames_pyramid, intermediate_outputs
        else:
            return predictions, residuals, gradients, confidences, predictions['final_flow'], residuals, original_frames_pyramid

class OpticalPIVNet(nn.Module):
    """Full gray-branch PIVNet: encoder + decoder."""
    def __init__(self, params, in_channels=2, num_layers=7):
        super().__init__()
        self.params = params
        self.encoder = Encoder(in_channels=in_channels, num_layers=num_layers)
        self.decoder = Decoder(params, num_layers=num_layers)
    
    def forward(self, x, binary_x=None, initial_binary_flow_high_res=None, return_intermediate=False):
        encoder_features, original_frames_pyramid, binary_frames_pyramid = self.encoder(x, binary_x=binary_x)
        if return_intermediate:
            predictions, residuals, gradients, confidences, final_flow, original_residuals, original_frames_pyramid, intermediate_outputs = self.decoder(
                encoder_features, original_frames_pyramid,
                initial_binary_flow_high_res=initial_binary_flow_high_res,
                return_intermediate=return_intermediate,
                binary_frames_pyramid=binary_frames_pyramid
            )
            return predictions, residuals, gradients, confidences, final_flow, original_residuals, original_frames_pyramid, intermediate_outputs
        else:
            predictions, residuals, gradients, confidences, final_flow, original_residuals, original_frames_pyramid = self.decoder(
                encoder_features, original_frames_pyramid,
                initial_binary_flow_high_res=initial_binary_flow_high_res,
                return_intermediate=return_intermediate,
                binary_frames_pyramid=binary_frames_pyramid
            )
            return predictions, residuals, gradients, confidences, final_flow, original_residuals, original_frames_pyramid

class BinaryCorrelationMatcher(nn.Module):
    """Binary correlation matcher with confidence-guided propagation."""
    def __init__(self, template_size=21, search_range=9):
        super().__init__()
        self.template_size = template_size
        self.search_range = search_range
        self.padding = template_size // 2
        sum_kernel = torch.ones(1, 1, template_size, template_size, dtype=torch.float32)
        self.register_buffer('sum_kernel', sum_kernel)
    
    def forward(self, p1, p2):
        p1 = p1.float()
        p2 = p2.float()
        B, _, H, W = p1.shape
        device = p1.device
        max_correlation_score = torch.full((B, 1, H, W), -1.0, device=device, dtype=torch.float32)
        initial_flow = torch.zeros((B, 2, H, W), device=device, dtype=torch.float32)
        p2_padded = F.pad(p2, (self.search_range, self.search_range, self.search_range, self.search_range), mode='constant', value=0)
        
        for dy in range(-self.search_range, self.search_range + 1):
            for dx in range(-self.search_range, self.search_range + 1):
                p2_shifted = p2_padded[:, :, self.search_range + dy : self.search_range + dy + H, self.search_range + dx : self.search_range + dx + W]
                agreement_map = (p1 * p2_shifted) + ((1 - p1) * (1 - p2_shifted))
                current_correlation = F.conv2d(agreement_map, self.sum_kernel, padding=self.padding, stride=1)
                update_mask = (current_correlation > max_correlation_score).squeeze(1)
                max_correlation_score[update_mask.unsqueeze(1)] = current_correlation[update_mask.unsqueeze(1)]
                initial_flow[:, 0][update_mask] = float(dx)
                initial_flow[:, 1][update_mask] = float(dy)
        
        particle_density = F.conv2d(p1, self.sum_kernel, padding=self.padding, stride=1)
        confidence = particle_density / (self.template_size * self.template_size)
        confidence = confidence.clamp(0.0, 1.0)
        return initial_flow, confidence

class PIVNet(nn.Module):
    """Full two-stream PIVNet wrapper."""
    def __init__(self, params):
        super().__init__()
        self.params = params
        template_size = getattr(params.model, 'binary_template_size', 21)
        search_range = getattr(params.model, 'binary_search_range', 9)
        print(f"Initializing BinaryCorrelationMatcher with template_size={template_size}, search_range={search_range}")
        self.binary_matcher = BinaryCorrelationMatcher(template_size=template_size, search_range=search_range)
        self.use_binary_initial_flow = getattr(params.model, 'binary_initial_flow', True)
        self.gray_branch = OpticalPIVNet(params, in_channels=params.model.in_channels, num_layers=7)
    
    def forward(self, gray_img_pair_batch, binary_img_pair_batch=None, precomputed_flow_b_hr=None, return_intermediate=False):
        _flow_b_hr = None
        if precomputed_flow_b_hr is not None:
            _flow_b_hr = precomputed_flow_b_hr
            if list(self.parameters()):
                model_device = next(self.parameters()).device
                if _flow_b_hr.device != model_device:
                    _flow_b_hr = _flow_b_hr.to(model_device)
        elif self.use_binary_initial_flow and binary_img_pair_batch is not None:
            with torch.no_grad():
                P1_binary = binary_img_pair_batch[:, 0:1, :, :]
                P2_binary = binary_img_pair_batch[:, 1:2, :, :]
                binary_match_output, _ = self.binary_matcher(P1_binary, P2_binary)
                _flow_b_hr = binary_match_output
        
        if return_intermediate:
            results = self.gray_branch(
                gray_img_pair_batch,
                binary_x=binary_img_pair_batch,
                initial_binary_flow_high_res=_flow_b_hr,
                return_intermediate=return_intermediate
            )
            return results
        else:
            results = self.gray_branch(
                gray_img_pair_batch,
                binary_x=binary_img_pair_batch,
                initial_binary_flow_high_res=_flow_b_hr,
                return_intermediate=return_intermediate
            )
            return results
