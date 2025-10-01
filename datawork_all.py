import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
import numpy as np
from PIL import Image
import json
import cv2
import torch.nn as nn
from piv_specific_augmentation import PIVAugmentation
import os


class MyTrainDataset(Dataset):
    def __init__(self, params_data_cfg, use_augmentation=True):
        """
        Initialize training dataset
        
        Args:
            params_data_cfg: Box object for data configuration (e.g., self.params.data from MyDataModule)
            use_augmentation: Whether to use data augmentation
        """
        self.data_root = getattr(params_data_cfg, 'data_root', '/path/to/data')
        self.data_list_path = getattr(params_data_cfg, 'train_data') 
        
        precomputed_dirname = getattr(params_data_cfg, 'precomputed_flow_b_hr_dirname_train', None)
        if precomputed_dirname is None:
            print("Warning: 'precomputed_flow_b_hr_dirname_train' not found in config, falling back to 'precomputed_flow_b_hr_dirname'.")
            precomputed_dirname = getattr(params_data_cfg, 'precomputed_flow_b_hr_dirname')
            
        self.precomputed_dir = precomputed_dirname
        print(f"MyTrainDataset: Using precomputed flow directory: {self.precomputed_dir}")

        self.data = []
        try:
            if self.data_list_path.endswith('.json'):
                with open(self.data_list_path, 'r') as f:
                    self.data = json.load(f)
                print(f"Successfully loaded training data (JSON): {len(self.data)} samples from {self.data_list_path}")
            else: 
                with open(self.data_list_path, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        items = line.strip().split()
                        if len(items) == 3: # [[img1_rel, img2_rel], flow_rel]
                            self.data.append([[items[0], items[1]], items[2]])
                print(f"Successfully loaded training data (LIST): {len(self.data)} samples from {self.data_list_path}")
        except Exception as e:
            print(f"Failed to load training data list {self.data_list_path}: {str(e)}")
            self.data = [] # Ensure self.data is initialized even on failure
            
        self.use_augmentation = use_augmentation
        if use_augmentation:
            self.augmenter = PIVAugmentation(prob=0.5, intensity=0.5) 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Ensure self.data is not empty to prevent errors if initialization failed
        if not self.data:
             print(f"Error: Training dataset is empty, cannot get item {index}")
             return ((torch.zeros(2, 256, 256), torch.zeros(2, 256, 256)), torch.zeros(2, 256, 256))

        try:
            data_item = self.data[index] 
            
            gray_img_orig = self.load_tif_file(data_item) 
            velocity_orig = self.load_flo_file(data_item)

            # Construct path for precomputed flow_b_hr
            # data_item[1] is the relative path of the original .flo file from data_root
            original_relative_flow_path = data_item[1]
            # The precomputed file will have the same relative path but with a .pt extension,
            # and will reside within self.precomputed_dir
            precomputed_file_relative_path = os.path.splitext(original_relative_flow_path)[0] + ".pt"
            precomputed_flow_path = os.path.join(self.precomputed_dir, precomputed_file_relative_path)

            if not os.path.exists(precomputed_flow_path):
                raise FileNotFoundError(f"Precomputed flow not found for item {index} at: {precomputed_flow_path}. Original flow rel path: {original_relative_flow_path}. Please ensure precomputation is complete.")
            
            precomputed_flow_b_hr = torch.load(precomputed_flow_path, map_location='cpu')

            if self.use_augmentation:
                binary_img_orig = convert_to_binary(gray_img_orig) 
                gray_img_aug, binary_img_aug, velocity_aug = self.augmenter(gray_img_orig, binary_img_orig, velocity_orig)
                return ((gray_img_aug, binary_img_aug, precomputed_flow_b_hr), velocity_aug)
            else:
                binary_img_orig = convert_to_binary(gray_img_orig)
                return ((gray_img_orig, binary_img_orig, precomputed_flow_b_hr), velocity_orig)

        except Exception as e:
            # Provide more context in error message if data_item is available
            flow_rel_path_for_error = data_item[1] if 'data_item' in locals() and data_item else 'N/A'
            print(f"Failed to load training sample {index} (flow_rel_path: {flow_rel_path_for_error}): {str(e)}")
            return ((torch.zeros(2, 256, 256), torch.zeros(2, 256, 256)), torch.zeros(2, 256, 256))

    def load_tif_file(self, data_item):
        img_paths = data_item[0]
        img_1_path = os.path.join(self.data_root, img_paths[0])
        img_2_path = os.path.join(self.data_root, img_paths[1])
        
        if not os.path.exists(img_1_path):
            raise FileNotFoundError(f"Image file does not exist: {img_1_path}")
        if not os.path.exists(img_2_path):
            raise FileNotFoundError(f"Image file does not exist: {img_2_path}")
            
        img_1 = torch.from_numpy(np.array(Image.open(img_1_path))/255.0).float()
        img_2 = torch.from_numpy(np.array(Image.open(img_2_path))/255.0).float()
        return torch.stack([img_1, img_2], dim=0)

    def load_flo_file(self, data_item):
        flow_path = os.path.join(self.data_root, data_item[1])
        
        if not os.path.exists(flow_path):
            raise FileNotFoundError(f"Flow field file does not exist: {flow_path}")
            
        with open(flow_path, 'rb') as f:
            magic = np.fromfile(f, np.float32, count=1)[0]
            if magic != 202021.25:
                raise ValueError(f"Invalid .flo file format: {flow_path}")
            width = np.fromfile(f, np.int32, count=1)[0]
            height = np.fromfile(f, np.int32, count=1)[0]
            data = np.fromfile(f, np.float32, count=2 * width * height)
            data = np.reshape(data, (height, width, 2))
        u = data[:, :, 0]
        v = data[:, :, 1]
        return torch.stack([torch.from_numpy(u).float(), torch.from_numpy(v).float()], dim=0)


class MyValDataset(Dataset): # Also used for Test
    def __init__(self, params_data_cfg, dataset_type='val'):
        """
        Initialize validation/test dataset
        Args:
            params_data_cfg: Box object for data configuration
            dataset_type: 'val' or 'test' to pick the correct list file from config
        """
        self.data_root = getattr(params_data_cfg, 'data_root', '/path/to/data')
        if dataset_type == 'val':
            self.data_list_path = getattr(params_data_cfg, 'val_data')
        elif dataset_type == 'test':
            self.data_list_path = getattr(params_data_cfg, 'test_data')
        else:
            raise ValueError(f"Invalid dataset_type: {dataset_type}. Must be 'val' or 'test'.")

        precomputed_dirname = getattr(params_data_cfg, 'precomputed_flow_b_hr_dirname_test')
        self.precomputed_dir = precomputed_dirname
        print(f"MyValDataset ({dataset_type}): Using precomputed flow directory: {self.precomputed_dir}")
        
        self.data = []
        try:
            if self.data_list_path.endswith('.json'):
                with open(self.data_list_path, 'r') as f:
                    self.data = json.load(f)
                print(f"Successfully loaded {dataset_type} data (JSON): {len(self.data)} samples from {self.data_list_path}")
            else: 
                with open(self.data_list_path, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        items = line.strip().split()
                        if len(items) == 3:
                            self.data.append([[items[0], items[1]], items[2]])
                print(f"Successfully loaded {dataset_type} data (LIST): {len(self.data)} samples from {self.data_list_path}")
        except Exception as e:
            print(f"Failed to load {dataset_type} data list {self.data_list_path}: {str(e)}")
            self.data = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if not self.data:
             print(f"Error: {self.dataset_type} dataset is empty, cannot get item {index}")
             return ((torch.zeros(2, 256, 256), torch.zeros(2, 256, 256)), torch.zeros(2, 256, 256))

        try:
            data_item = self.data[index]
            gray_img = self.load_tif_file(data_item)
            velocity = self.load_flo_file(data_item)
            binary_img = convert_to_binary(gray_img) # Generate binary image from grayscale

            original_relative_flow_path = data_item[1]
            precomputed_file_relative_path = os.path.splitext(original_relative_flow_path)[0] + ".pt"
            precomputed_flow_path = os.path.join(self.precomputed_dir, precomputed_file_relative_path)

            if not os.path.exists(precomputed_flow_path):
                 raise FileNotFoundError(f"Precomputed flow not found for item {index} at: {precomputed_flow_path}. Original flow rel path: {original_relative_flow_path}. Please ensure precomputation is complete.")
            
            precomputed_flow_b_hr = torch.load(precomputed_flow_path, map_location='cpu')
            
            return ((gray_img, binary_img, precomputed_flow_b_hr), velocity)

        except Exception as e:
            flow_rel_path_for_error = data_item[1] if 'data_item' in locals() and data_item else 'N/A'
            print(f"Failed to load validation/test sample {index} (flow_rel_path: {flow_rel_path_for_error}): {str(e)}")
            return ((torch.zeros(2, 256, 256), torch.zeros(2, 256, 256)), torch.zeros(2, 256, 256))

    def load_tif_file(self, data_item):
        img_paths = data_item[0]
        img_1_path = os.path.join(self.data_root, img_paths[0])
        img_2_path = os.path.join(self.data_root, img_paths[1])
        if not os.path.exists(img_1_path): raise FileNotFoundError(f"Image file does not exist: {img_1_path}")
        if not os.path.exists(img_2_path): raise FileNotFoundError(f"Image file does not exist: {img_2_path}")
        img_1 = torch.from_numpy(np.array(Image.open(img_1_path))/255.0).float()
        img_2 = torch.from_numpy(np.array(Image.open(img_2_path))/255.0).float()
        return torch.stack([img_1, img_2], dim=0)

    def load_flo_file(self, data_item):
        flow_path = os.path.join(self.data_root, data_item[1])
        if not os.path.exists(flow_path): raise FileNotFoundError(f"Flow field file does not exist: {flow_path}")
        with open(flow_path, 'rb') as f:
            magic = np.fromfile(f, np.float32, count=1)[0]
            if magic != 202021.25: raise ValueError(f"Invalid .flo file format: {flow_path}")
            width = np.fromfile(f, np.int32, count=1)[0]
            height = np.fromfile(f, np.int32, count=1)[0]
            data = np.fromfile(f, np.float32, count=2 * width * height)
            data = np.reshape(data, (height, width, 2))
        u = data[:, :, 0]
        v = data[:, :, 1]
        return torch.stack([torch.from_numpy(u).float(), torch.from_numpy(v).float()], dim=0)

def convert_to_binary(gray_img, threshold=0.5):
    """Convert grayscale image to binary using threshold"""
    binary_img = (gray_img > threshold).float()
    return binary_img

class MyDataModule(LightningDataModule):
    """PyTorch Lightning DataModule for PIV data"""
    def __init__(self, params): # params is the full config Box object
        super().__init__()
        self.params = params 
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        data_cfg = self.params.data 
        
        if stage == 'fit' or stage is None:
            self.train_dataset = MyTrainDataset(
                params_data_cfg=data_cfg,
                use_augmentation=getattr(data_cfg, 'use_augmentation', True)
            )
            self.val_dataset = MyValDataset(
                params_data_cfg=data_cfg,
                dataset_type='val'
            )
        if stage == 'test' or stage is None:
            self.test_dataset = MyValDataset(
                params_data_cfg=data_cfg,
                dataset_type='test'
            )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.params.training.batch_size,
                        shuffle=True, num_workers=self.params.training.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.params.training.batch_size,
                        shuffle=False, num_workers=self.params.training.num_workers
                        )

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.params.training.batch_size,
                        shuffle=False, num_workers=self.params.training.num_workers)
