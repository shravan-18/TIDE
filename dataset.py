import os
import glob
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch


class UnderwaterDataset(Dataset):
    """Dataset for underwater image restoration"""
    def __init__(self, root_dir, subset='train', img_size=256, crop_size=None, augment=True):
        """
        Args:
            root_dir: Root directory of the dataset
            subset: 'train' or 'valid'
            img_size: Size to resize images to
            crop_size: Size to randomly crop images to during training (if None, don't crop)
            augment: Whether to apply data augmentation
        """
        self.root_dir = root_dir
        self.subset = subset
        self.img_size = img_size
        self.crop_size = crop_size if crop_size is not None else img_size
        self.augment = augment and subset == 'train'  # Only augment training data
        
        # Define paths
        self.degraded_dir = os.path.join(root_dir, f"{subset}_data/degraded")
        self.reference_dir = os.path.join(root_dir, f"{subset}_data/reference")
        
        # Get image files
        image_extensions = ["*.png", "*.PNG", "*.jpg", "*.JPG", "*.jpeg", "*.JPEG", "*.bmp", "*.BMP", "*.tiff", "*.TIFF", "*.webp", "*.WEBP"]

        # Collect all degraded image files
        self.degraded_files = []
        for ext in image_extensions:
            self.degraded_files.extend(glob.glob(os.path.join(self.degraded_dir, ext)))
        self.degraded_files = sorted(self.degraded_files)

        # Collect all reference image files
        self.reference_files = []
        for ext in image_extensions:
            self.reference_files.extend(glob.glob(os.path.join(self.reference_dir, ext)))
        self.reference_files = sorted(self.reference_files) 

        
        # Verify that degraded and reference files match
        self._verify_files()
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        print(f"Loaded {len(self.degraded_files)} {subset} image pairs")
        
    def _verify_files(self):
        """Verify that degraded and reference files match"""
        degraded_names = [os.path.basename(f) for f in self.degraded_files]
        reference_names = [os.path.basename(f) for f in self.reference_files]
        
        # Check if the number of files match
        if len(degraded_names) != len(reference_names):
            raise ValueError(f"Number of degraded ({len(degraded_names)}) and reference ({len(reference_names)}) images don't match")
        
        # Check if the filenames match
        if set(degraded_names) != set(reference_names):
            mismatch = set(degraded_names).symmetric_difference(set(reference_names))
            raise ValueError(f"Mismatch in filenames: {mismatch}")
        
    def __len__(self):
        return len(self.degraded_files)
        
    def _load_and_transform_image(self, image_path):
        """Load an image and apply transformations"""
        image = Image.open(image_path).convert('RGB')
        
        # Resize to target size
        image = image.resize((self.img_size, self.img_size), Image.LANCZOS)
        
        return image
        
    def __getitem__(self, idx):
        # Get image paths
        degraded_path = self.degraded_files[idx]
        reference_path = self.reference_files[idx]
        
        # Load images
        degraded_img = self._load_and_transform_image(degraded_path)
        reference_img = self._load_and_transform_image(reference_path)
        
        # Apply augmentations
        if self.augment:
            # Random crop
            if self.crop_size < self.img_size:
                i, j, h, w = transforms.RandomCrop.get_params(
                    degraded_img, output_size=(self.crop_size, self.crop_size))
                degraded_img = transforms.functional.crop(degraded_img, i, j, h, w)
                reference_img = transforms.functional.crop(reference_img, i, j, h, w)
            
            # Random horizontal and vertical flips
            if random.random() > 0.5:
                degraded_img = transforms.functional.hflip(degraded_img)
                reference_img = transforms.functional.hflip(reference_img)
                
            if random.random() > 0.5:
                degraded_img = transforms.functional.vflip(degraded_img)
                reference_img = transforms.functional.vflip(reference_img)
            
            # Random rotation
            if random.random() > 0.5:
                angle = random.choice([90, 180, 270])
                degraded_img = transforms.functional.rotate(degraded_img, angle)
                reference_img = transforms.functional.rotate(reference_img, angle)
                
            # Color jitter (only for degraded image, as reference should be clean)
            if random.random() > 0.7:
                brightness = 0.1
                contrast = 0.1
                saturation = 0.1
                hue = 0.05
                
                color_jitter = transforms.ColorJitter(
                    brightness=brightness,
                    contrast=contrast,
                    saturation=saturation,
                    hue=hue
                )
                degraded_img = color_jitter(degraded_img)
        
        # Convert to tensors
        degraded_tensor = self.transform(degraded_img)
        reference_tensor = self.transform(reference_img)
        
        return {
            'degraded': degraded_tensor,
            'reference': reference_tensor,
            'filename': os.path.basename(degraded_path)
        }


def get_dataloaders(root_dir, batch_size=16, img_size=256, crop_size=None, 
                    num_workers=4, augment=True):
    """Create training and validation dataloaders"""
    
    # Create datasets
    train_dataset = UnderwaterDataset(
        root_dir=root_dir,
        subset='train',
        img_size=img_size,
        crop_size=crop_size,
        augment=augment
    )
    
    val_dataset = UnderwaterDataset(
        root_dir=root_dir,
        subset='valid',
        img_size=img_size,
        crop_size=None,  # No random cropping for validation
        augment=False    # No augmentation for validation
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader
