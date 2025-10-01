import torch
import torch.nn.functional as F

#############################
# SSIM helper function
#############################
def ssim(x, y, c1=1e-4, c2=9e-4, kernel_size=3, stride=1):
    """
    Compute differentiable SSIM between two images.
    Args:
        x, y: [B, C, H, W] input images
    Returns:
        mean SSIM value
    """
    pool = F.avg_pool2d
    x = x.float()
    y = y.float()
    mu_x = pool(x, kernel_size, stride)
    mu_y = pool(y, kernel_size, stride)
    sigma_x = pool(x * x, kernel_size, stride) - mu_x * mu_x
    sigma_y = pool(y * y, kernel_size, stride) - mu_y * mu_y
    sigma_xy = pool(x * y, kernel_size, stride) - mu_x * mu_y
    sigma_x = torch.clamp(sigma_x, min=0.0)
    sigma_y = torch.clamp(sigma_y, min=0.0)
    ssim_num = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    ssim_den = (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
    ssim_den = torch.clamp(ssim_den, min=1e-8)
    ssim_map = ssim_num / ssim_den
    return torch.clamp(ssim_map.mean(), -1, 1)

#############################
# Robust photometric warp loss
#############################
def compute_robust_photometric_loss(predicted_flow, frame1, frame2, alpha=0.85):
    """
    Compute robust photometric warp loss using L1 and SSIM.
    Args:
        predicted_flow: [B, 2, H, W]
        frame1, frame2: [B, 1, H, W]
        alpha: L1/SSIM weight
    Returns:
        Combined L1 + SSIM loss
    """
    predicted_flow = predicted_flow.float()
    frame1 = frame1.float()
    frame2 = frame2.float()
    b, _, h, w = predicted_flow.shape
    device = predicted_flow.device
    max_disp = min(h, w) * 0.5
    predicted_flow = torch.clamp(predicted_flow, -max_disp, max_disp)
    xx = torch.linspace(-1.0, 1.0, w, device=device, dtype=frame1.dtype).view(1, 1, 1, w).expand(b, -1, h, -1)
    yy = torch.linspace(-1.0, 1.0, h, device=device, dtype=frame1.dtype).view(1, 1, h, 1).expand(b, -1, -1, w)
    base_grid = torch.cat((xx, yy), 1)
    norm_flow_u = predicted_flow[:, 0:1, :, :] / ((w - 1.0) / 2.0)
    norm_flow_v = predicted_flow[:, 1:2, :, :] / ((h - 1.0) / 2.0)
    norm_flow = torch.cat((norm_flow_u, norm_flow_v), dim=1)
    norm_flow = torch.clamp(norm_flow, -2.0, 2.0)
    sampling_grid = (base_grid + norm_flow).permute(0, 2, 3, 1)
    warped_frame1 = F.grid_sample(frame1, sampling_grid, mode='bilinear', padding_mode='border', align_corners=True)
    l1_loss = F.l1_loss(warped_frame1, frame2)
    ssim_loss = (1 - ssim(warped_frame1, frame2)) / 2.0
    robust_loss = alpha * l1_loss + (1 - alpha) * ssim_loss
    return robust_loss

#############################
# Photometric warp loss (MSE)
#############################
def compute_photometric_warp_loss(predicted_flow, frame1, frame2):
    """
    Compute photometric warp loss (MSE) between frame2 and frame1 warped by predicted_flow.
    Args:
        predicted_flow: [B, 2, H, W]
        frame1, frame2: [B, 1, H, W]
    Returns:
        MSE loss
    """
    predicted_flow = predicted_flow.float()
    frame1 = frame1.float()
    frame2 = frame2.float()
    b, _, h, w = predicted_flow.shape
    device = predicted_flow.device
    xx = torch.linspace(-1.0, 1.0, w, device=device, dtype=frame1.dtype).view(1, 1, 1, w).expand(b, -1, h, -1)
    yy = torch.linspace(-1.0, 1.0, h, device=device, dtype=frame1.dtype).view(1, 1, h, 1).expand(b, -1, -1, w)
    base_grid = torch.cat((xx, yy), 1)
    norm_flow_u = predicted_flow[:, 0:1, :, :] / ((w - 1.0) / 2.0)
    norm_flow_v = predicted_flow[:, 1:2, :, :] / ((h - 1.0) / 2.0)
    norm_flow = torch.cat((norm_flow_u, norm_flow_v), dim=1)
    sampling_grid = (base_grid + norm_flow).permute(0, 2, 3, 1)
    warped_frame1 = F.grid_sample(frame1, sampling_grid, mode='bilinear', padding_mode='border', align_corners=True)
    loss = F.mse_loss(warped_frame1, frame2)
    return loss

#############################
# Optical constraint loss
#############################
def optical_loss(pred_residual, gradients, confidence=None):
    """
    Optical flow constraint loss: Ix*Δu + Iy*Δv + It = 0
    Args:
        pred_residual: [B,2,H,W]
        gradients: dict with keys 'Ix', 'Iy', 'It'
        confidence: [B,1,H,W] or None
    Returns:
        Weighted MSE of optical flow constraint residual
    """
    # Gradient clipping for numerical stability
    ix = torch.clamp(gradients['Ix'], -0.5, 0.5)
    iy = torch.clamp(gradients['Iy'], -0.5, 0.5)
    it = torch.clamp(gradients['It'], -0.5, 0.5)
    
    # Residual clipping to prevent large values
    du = torch.clamp(pred_residual[:, 0:1, :, :], -50.0, 50.0)
    dv = torch.clamp(pred_residual[:, 1:2, :, :], -50.0, 50.0)
    
    residual = ix * du + iy * dv + it
    zero_target = torch.zeros_like(residual)
    
    if confidence is not None:
        # Enhanced numerical stability
        confidence_sum = confidence.sum() + 1e-8
        weighted_loss = ((residual - zero_target) ** 2 * confidence).sum() / confidence_sum
        return weighted_loss
    else:
        return F.mse_loss(residual, zero_target)

def compute_spatially_weighted_smoothness_loss(pred_flow, weight_map):
    """
    Spatially weighted L2 smoothness loss.
    Args:
        pred_flow: [B,2,H,W]
        weight_map: [B,1,H,W]
    Returns:
        Weighted smoothness loss
    """
    flow_w_grad = pred_flow[:, :, :, 1:] - pred_flow[:, :, :, :-1]
    w_loss_map = torch.sum(flow_w_grad ** 2, dim=1, keepdim=True)
    weight_map_w = weight_map[:, :, :, :-1]
    weighted_w_loss = (w_loss_map * weight_map_w).sum() / (weight_map_w.sum() + 1e-6)
    flow_h_grad = pred_flow[:, :, 1:, :] - pred_flow[:, :, :-1, :]
    h_loss_map = torch.sum(flow_h_grad ** 2, dim=1, keepdim=True)
    weight_map_h = weight_map[:, :, :-1, :]
    weighted_h_loss = (h_loss_map * weight_map_h).sum() / (weight_map_h.sum() + 1e-6)
    return weighted_w_loss + weighted_h_loss

def compute_spatially_weighted_continuity_loss(pred_flow, weight_map):
    """
    Spatially weighted continuity loss (divergence).
    Args:
        pred_flow: [B,2,H,W]
        weight_map: [B,1,H,W]
    Returns:
        Weighted continuity loss
    """
    u = pred_flow[:, 0:1, :, :]
    v = pred_flow[:, 1:2, :, :]
    kernel_dx = torch.tensor([[[[-0.5, 0, 0.5]]]], device=pred_flow.device, dtype=pred_flow.dtype)
    kernel_dy = torch.tensor([[[[-0.5], [0], [0.5]]]], device=pred_flow.device, dtype=pred_flow.dtype)
    u_padded = F.pad(u, (1, 1, 0, 0), mode='replicate')
    v_padded = F.pad(v, (0, 0, 1, 1), mode='replicate')
    du_dx = F.conv2d(u_padded, kernel_dx, stride=1)
    dv_dy = F.conv2d(v_padded, kernel_dy, stride=1)
    div = du_dx + dv_dy
    loss_map = div ** 2
    weighted_loss_map = loss_map * weight_map
    return weighted_loss_map.sum() / (weight_map.sum() + 1e-6)

#############################
# Edge-aware smoothness loss
#############################
def compute_edge_aware_smoothness_loss(pred_flow, frame1, weight_map=None):
    """
    Edge-aware smoothness loss: less penalty on edges.
    Args:
        pred_flow: [B,2,H,W]
        frame1: [B,1,H,W]
        weight_map: [B,1,H,W] (optional)
    Returns:
        Edge-aware smoothness loss
    """
    sobel_x = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], device=pred_flow.device, dtype=pred_flow.dtype) / 8.0
    sobel_y = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], device=pred_flow.device, dtype=pred_flow.dtype) / 8.0
    ix = F.conv2d(frame1, sobel_x, padding=1)
    iy = F.conv2d(frame1, sobel_y, padding=1)
    edge_magnitude = torch.sqrt(ix ** 2 + iy ** 2)
    edge_magnitude = torch.clamp(edge_magnitude, 0, 1)
    edge_aware_weight = 1.0 - edge_magnitude
    flow_w_grad = pred_flow[:, :, :, 1:] - pred_flow[:, :, :, :-1]
    flow_h_grad = pred_flow[:, :, 1:, :] - pred_flow[:, :, :-1, :]
    edge_aware_weight_w = edge_aware_weight[:, :, :, :-1]
    edge_aware_weight_h = edge_aware_weight[:, :, :-1, :]
    w_loss = torch.sum(flow_w_grad ** 2, dim=1, keepdim=True) * edge_aware_weight_w
    h_loss = torch.sum(flow_h_grad ** 2, dim=1, keepdim=True) * edge_aware_weight_h
    if weight_map is not None:
        weight_map_w = weight_map[:, :, :, :-1]
        weight_map_h = weight_map[:, :, :-1, :]
        w_loss = w_loss * weight_map_w
        h_loss = h_loss * weight_map_h
    total_weight_w = edge_aware_weight_w.sum() + 1e-6
    total_weight_h = edge_aware_weight_h.sum() + 1e-6
    return (w_loss.sum() / total_weight_w + h_loss.sum() / total_weight_h) / 2.0

#############################
# Multiscale consistency loss
#############################
def compute_multiscale_consistency_loss(predictions, scale_weights):
    """
    Multiscale consistency loss: ensure consistency between different scales.
    Args:
        predictions: dict of flow predictions at each scale
        scale_weights: dict of weights for each scale
    Returns:
        Multiscale consistency loss
    """
    consistency_loss = torch.tensor(0.0, device=next(iter(predictions.values())).device, dtype=next(iter(predictions.values())).dtype)
    flow_levels = sorted([k for k in predictions.keys() if k.startswith('flow_level_')], key=lambda x: int(x.split('_')[-1]))
    for i in range(len(flow_levels) - 1):
        current_level = flow_levels[i]
        next_level = flow_levels[i + 1]
        current_flow = predictions[current_level]
        next_flow = predictions[next_level]
        upsampled_next_flow = F.interpolate(next_flow, size=current_flow.shape[2:], mode='bilinear', align_corners=True)
        scale_factor = current_flow.shape[2] / next_flow.shape[2]
        upsampled_next_flow = upsampled_next_flow * scale_factor
        consistency = F.mse_loss(current_flow, upsampled_next_flow)
        current_weight = scale_weights.get(current_level, 0.1)
        consistency_loss += current_weight * consistency
    return consistency_loss

#############################
# Main total loss for gray branch (multi-scale)
#############################
def compute_gray_total_loss(predictions, residuals, original_residuals, gradients, confidences, gt_flow, loss_weights,
                            unsupervised_mode=False, original_frames_pyramid=None, current_epoch=0, 
                            unsupervised_settings=None):
    """
    Compute total multi-scale loss for gray branch.
    Args:
        predictions, residuals, original_residuals, gradients, confidences: dicts from model
        gt_flow: ground truth flow
        loss_weights: Box/dict of weights
        unsupervised_mode: bool
        original_frames_pyramid: list of [B,2,H,W] (for unsupervised)
        current_epoch: int
        unsupervised_settings: dict
    Returns:
        total_loss: scalar
        loss_detail: dict
    """
    total_loss = torch.tensor(0.0, device=next(iter(predictions.values())).device, dtype=next(iter(predictions.values())).dtype)
    loss_detail = {}
    flow_levels = sorted([k for k in predictions.keys() if k.startswith('flow_level_')], key=lambda x: int(x.split('_')[-1]), reverse=True)
    
    # Dynamic warp weight scheduling for unsupervised training
    if unsupervised_mode and unsupervised_settings and unsupervised_settings.get('warp_weight_schedule', False):
        initial_warp_weight = unsupervised_settings.get('initial_warp_weight', 20.0)
        final_warp_weight = unsupervised_settings.get('final_warp_weight', 5.0)
        decay_epochs = unsupervised_settings.get('warp_weight_decay_epochs', 1000)
        if current_epoch < decay_epochs:
            progress = current_epoch / decay_epochs
            current_warp_weight = initial_warp_weight + (final_warp_weight - initial_warp_weight) * progress
        else:
            current_warp_weight = final_warp_weight
    else:
        current_warp_weight = loss_weights['warp']
    
    for level_key in flow_levels:
        level_idx = int(level_key.split('_')[-1])
        curr_residual = residuals[level_key]
        curr_original_residual = original_residuals[level_key]
        curr_prediction = predictions[level_key]
        curr_gradients = gradients[f'level_{level_idx}']
        curr_confidence = confidences[f'level_{level_idx}']
        
        # Optical flow constraint loss
        loss_opt = optical_loss(curr_residual, curr_gradients, curr_confidence)
        
        # Smoothness and continuity losses
        smoothness_weight_map = curr_confidence
        if unsupervised_mode:
            if unsupervised_settings and unsupervised_settings.get('use_edge_aware_smoothness', False):
                images_level = original_frames_pyramid[level_idx]
                frame1_level = images_level[:, 0:1, :, :]
                loss_sm = compute_edge_aware_smoothness_loss(curr_prediction, frame1_level, smoothness_weight_map)
            else:
                loss_sm = compute_spatially_weighted_smoothness_loss(curr_prediction, smoothness_weight_map)
            loss_cont = compute_spatially_weighted_continuity_loss(curr_prediction, smoothness_weight_map)
        else:
            # Supervised mode
            loss_sm = compute_spatially_weighted_smoothness_loss(curr_prediction, smoothness_weight_map)
            loss_cont = compute_spatially_weighted_continuity_loss(curr_prediction, smoothness_weight_map)
        
        loss_detail[f'{level_key}_optical'] = loss_opt.item()
        loss_detail[f'{level_key}_smooth'] = loss_sm.item()
        loss_detail[f'{level_key}_cont'] = loss_cont.item()
        
        scale_w = loss_weights['scale_weights'].get(level_key, 0.1)
        level_loss = scale_w * (
            loss_weights['optical'] * loss_opt +
            loss_weights['smooth'] * loss_sm +
            loss_weights['cont'] * loss_cont
        )
        
        if unsupervised_mode:
            if original_frames_pyramid is None:
                raise ValueError("original_frames_pyramid must be provided for unsupervised mode.")
            images_level = original_frames_pyramid[level_idx]
            frame1_level = images_level[:, 0:1, :, :]
            frame2_level = images_level[:, 1:2, :, :]
            scale_warp_loss = compute_robust_photometric_loss(curr_prediction, frame1_level, frame2_level)
            loss_detail[f'{level_key}_warp'] = scale_warp_loss.item()
            level_loss += scale_w * current_warp_weight * scale_warp_loss
        else:
            # Data loss for supervised mode
            data_weight = loss_weights['data']
            gt_flow_down = F.interpolate(gt_flow, size=curr_prediction.shape[2:], mode='bilinear', align_corners=True)
            scale_flow_loss = F.mse_loss(curr_prediction, gt_flow_down)
            loss_detail[f'{level_key}_data'] = scale_flow_loss.item()
            level_loss += scale_w * data_weight * scale_flow_loss
        
        total_loss += level_loss
    
    # Multiscale consistency loss for unsupervised mode
    if unsupervised_mode and unsupervised_settings and unsupervised_settings.get('use_multiscale_consistency', False):
        consistency_loss = compute_multiscale_consistency_loss(predictions, loss_weights['scale_weights'])
        consistency_weight = 0.1
        total_loss += consistency_weight * consistency_loss
        loss_detail['multiscale_consistency'] = consistency_loss.item()
    
    return total_loss, loss_detail
