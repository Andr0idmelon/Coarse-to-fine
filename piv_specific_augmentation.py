import torch
import torch.nn.functional as F
import numpy as np
import random

class PIVAugmentation:
    """用于PIV图像的特定数据增强"""
    
    def __init__(self, prob=0.5, intensity=0.5, intensity_scale=0.5):
        """
        初始化PIV增强器
        :param prob: 应用每种增强的概率
        :param intensity: 增强的强度 (0-1)
        """
        self.prob = prob
        self.intensity = intensity * intensity_scale 
    
    def __call__(self, gray_img_pair, binary_img_pair, flow):
        """
        应用随机增强到PIV图像对和对应的流场
        """
        enhanced_gray = gray_img_pair.clone()
        enhanced_binary = binary_img_pair.clone()
        enhanced_flow = flow.clone()
        
        if random.random() < self.prob:
            enhanced_gray, enhanced_binary, enhanced_flow = self.particle_density_variation(
                enhanced_gray, enhanced_binary, enhanced_flow)
        
        if random.random() < self.prob:
            enhanced_gray = self.brightness_contrast_variation(enhanced_gray)
        
        if random.random() < self.prob:
            enhanced_gray = self.add_noise(enhanced_gray)
        
        if random.random() < self.prob:
            enhanced_gray = self.background_variation(enhanced_gray)
        
        if random.random() < self.prob:
            enhanced_gray, enhanced_flow = self.laser_sheet_fluctuation(
                enhanced_gray, enhanced_flow)
        
        return enhanced_gray, enhanced_binary, enhanced_flow
    
    def particle_density_variation(self, gray_img, binary_img, flow):
        """模拟粒子密度变化"""
        if len(gray_img.shape) == 3:  
            gray_img = gray_img.unsqueeze(0)  
            binary_img = binary_img.unsqueeze(0)
            flow = flow.unsqueeze(0)
            is_batch = False
        else:
            is_batch = True
            
        batch_size = gray_img.shape[0]
        for b in range(batch_size):
            ratio = self.intensity * 0.3 * random.uniform(-1, 1)  
            
            particles = binary_img[b, 0] > 0.5
            
            if ratio > 0:  
                non_particles = ~particles
                non_particle_indices = torch.nonzero(non_particles)
                
                if len(non_particle_indices) > 0:
                    num_to_add = int(particles.sum() * ratio)
                    indices_to_add = non_particle_indices[torch.randperm(len(non_particle_indices))[:num_to_add]]
                    
                    for idx in indices_to_add:
                        y, x = idx[0], idx[1]
                        gray_img[b, 0, y, x] = random.uniform(0.8, 1.0)  
                        binary_img[b, 0, y, x] = 1.0
                        
                        flow_y, flow_x = flow[b, 1, y, x].item(), flow[b, 0, y, x].item()
                        new_y, new_x = int(y + flow_y), int(x + flow_x)
                        
                        h, w = gray_img.shape[2], gray_img.shape[3]
                        if 0 <= new_y < h and 0 <= new_x < w:
                            gray_img[b, 1, new_y, new_x] = random.uniform(0.8, 1.0)
                            binary_img[b, 1, new_y, new_x] = 1.0
            
            else:  
                ratio = -ratio  
                particle_indices = torch.nonzero(particles)
                
                if len(particle_indices) > 0:
                    num_to_remove = int(particles.sum() * ratio)
                    indices_to_remove = particle_indices[torch.randperm(len(particle_indices))[:num_to_remove]]
                    
                    for idx in indices_to_remove:
                        y, x = idx[0], idx[1]
                        gray_img[b, 0, y, x] = random.uniform(0.0, 0.2)  
                        binary_img[b, 0, y, x] = 0.0
                        
                        flow_y, flow_x = flow[b, 1, y, x].item(), flow[b, 0, y, x].item()
                        new_y, new_x = int(y + flow_y), int(x + flow_x)
                        
                        h, w = gray_img.shape[2], gray_img.shape[3]
                        if 0 <= new_y < h and 0 <= new_x < w:
                            gray_img[b, 1, new_y, new_x] = random.uniform(0.0, 0.2)
                            binary_img[b, 1, new_y, new_x] = 0.0
        
        if not is_batch:
            gray_img = gray_img.squeeze(0)
            binary_img = binary_img.squeeze(0)
            flow = flow.squeeze(0)
                            
        return gray_img, binary_img, flow
    
    def brightness_contrast_variation(self, gray_img):
        """模拟亮度和对比度变化"""
        if len(gray_img.shape) == 3:  
            gray_img = gray_img.unsqueeze(0)  
            is_batch = False
        else:
            is_batch = True
            
        batch_size = gray_img.shape[0]
        for b in range(batch_size):
            brightness_factor = 1.0 + self.intensity * random.uniform(-0.6, 0.6)
            contrast_factor = 1.0 + self.intensity * random.uniform(-0.6, 0.6)
            
            for i in range(2):
                img = gray_img[b, i] * brightness_factor
                
                mean = torch.mean(img)
                img = (img - mean) * contrast_factor + mean
                
                gray_img[b, i] = torch.clamp(img, 0.0, 1.0)
        
        if not is_batch:
            gray_img = gray_img.squeeze(0)
                
        return gray_img
    
    def add_noise(self, gray_img):
        """添加高斯噪声"""
        if len(gray_img.shape) == 3:  
            gray_img = gray_img.unsqueeze(0)  
            is_batch = False
        else:
            is_batch = True
            
        batch_size = gray_img.shape[0]
        for b in range(batch_size):
            noise_std = self.intensity * 0.15 * random.uniform(0.5, 1.5)
            
            for i in range(2):
                noise = torch.randn_like(gray_img[b, i]) * noise_std
                gray_img[b, i] = torch.clamp(gray_img[b, i] + noise, 0.0, 1.0)
        
        if not is_batch:
            gray_img = gray_img.squeeze(0)
                
        return gray_img
    
    def background_variation(self, gray_img):
        """模拟背景不均匀性"""
        if len(gray_img.shape) == 3:  
            gray_img = gray_img.unsqueeze(0)  
            is_batch = False
        else:
            is_batch = True
            
        batch_size = gray_img.shape[0]
        h, w = gray_img.shape[2], gray_img.shape[3]
        
        for b in range(batch_size):
            variation_amp = self.intensity * 0.25 * random.uniform(0.5, 1.5)
            
            y_grid, x_grid = torch.meshgrid(
                torch.linspace(-1, 1, h),
                torch.linspace(-1, 1, w),
                indexing='ij'
            )
            
            freq_x = random.uniform(1, 3)
            freq_y = random.uniform(1, 3)
            phase_x = random.uniform(0, 2 * np.pi)
            phase_y = random.uniform(0, 2 * np.pi)
            
            for i in range(2):
                background = (torch.sin(freq_x * x_grid + phase_x) + 
                              torch.sin(freq_y * y_grid + phase_y)) / 2.0
                              
                background = background * variation_amp
                
                gray_img[b, i] = torch.clamp(gray_img[b, i] + background, 0.0, 1.0)
        
        if not is_batch:
            gray_img = gray_img.squeeze(0)
            
        return gray_img
    
    def laser_sheet_fluctuation(self, gray_img, flow):
        """模拟激光平面波动，轻微变形图像"""
        if len(gray_img.shape) == 3:  
            gray_img = gray_img.unsqueeze(0)  
            flow = flow.unsqueeze(0)  
            is_batch = False
        else:
            is_batch = True
            
        batch_size = gray_img.shape[0]
        h, w = gray_img.shape[2], gray_img.shape[3]
        
        for b in range(batch_size):
            amplitude = self.intensity * 4.0
            
            y_grid, x_grid = torch.meshgrid(
                torch.arange(h, dtype=torch.float32),
                torch.arange(w, dtype=torch.float32),
                indexing='ij'
            )
            grid = torch.stack([x_grid, y_grid], dim=0).to(gray_img.device)
            
            distortion_x = amplitude * torch.sin(2 * np.pi * y_grid / h * random.uniform(1, 3))
            distortion_y = amplitude * torch.sin(2 * np.pi * x_grid / w * random.uniform(1, 3))
            
            x_new = torch.clamp(x_grid + distortion_x, 0, w-1)
            y_new = torch.clamp(y_grid + distortion_y, 0, h-1)
            
            grid_sample = torch.stack([
                2.0 * x_new / (w - 1) - 1.0,
                2.0 * y_new / (h - 1) - 1.0
            ], dim=-1).unsqueeze(0)  # [1,H,W,2]
            
            gray_img[b, 1:2] = F.grid_sample(
                gray_img[b, 1:2].unsqueeze(0),
                grid_sample,
                mode='bilinear',
                padding_mode='border',
                align_corners=True
            ).squeeze(0)
            
            flow_x = flow[b, 0:1].unsqueeze(0)
            flow_y = flow[b, 1:2].unsqueeze(0)
            
            flow_x = F.grid_sample(
                flow_x, grid_sample, mode='bilinear', 
                padding_mode='border', align_corners=True
            ).squeeze(0)
            
            flow_y = F.grid_sample(
                flow_y, grid_sample, mode='bilinear', 
                padding_mode='border', align_corners=True
            ).squeeze(0)
            
            flow_x = flow_x + distortion_x.unsqueeze(0)
            flow_y = flow_y + distortion_y.unsqueeze(0)
            
            flow[b, 0:1] = flow_x
            flow[b, 1:2] = flow_y
        
        if not is_batch:
            gray_img = gray_img.squeeze(0)
            flow = flow.squeeze(0)
            
        return gray_img, flow 