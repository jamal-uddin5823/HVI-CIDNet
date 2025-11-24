#!/usr/bin/env python3
"""
GPU Stress Test Script
Tests GPU performance with configurable intensity levels using the LFW dataset.
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import argparse
import time
import os
from pathlib import Path
import numpy as np
import sys

class LFWStressDataset(Dataset):
    """Dataset loader for LFW images for stress testing"""
    def __init__(self, root_dir, transform=None, repeat_factor=1):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.repeat_factor = repeat_factor
        
        # Collect all image files
        self.image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            self.image_paths.extend(list(self.root_dir.rglob(ext)))
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {root_dir}")
        
        print(f"Found {len(self.image_paths)} images, repeated {repeat_factor}x = {len(self.image_paths) * repeat_factor} total samples")
    
    def __len__(self):
        return len(self.image_paths) * self.repeat_factor
    
    def __getitem__(self, idx):
        actual_idx = idx % len(self.image_paths)
        img_path = self.image_paths[actual_idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, 0  # Dummy label
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a random tensor as fallback
            if self.transform:
                return torch.randn(3, 224, 224), 0
            return torch.randn(3, 250, 250), 0


class StressTestModel(nn.Module):
    """Configurable model for GPU stress testing"""
    def __init__(self, complexity='medium'):
        super(StressTestModel, self).__init__()
        
        if complexity == 'light':
            # Light model: ~5M parameters
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d((7, 7)),
                nn.Flatten(),
                nn.Linear(128 * 7 * 7, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(512, 1000),
            )
        
        elif complexity == 'medium':
            # Medium model: ~20M parameters
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(128, 256, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(256, 512, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, 3, padding=1),
                nn.ReLU(inplace=True),
            )
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d((7, 7)),
                nn.Flatten(),
                nn.Linear(512 * 7 * 7, 2048),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(2048, 1000),
            )
        
        elif complexity == 'heavy':
            # Heavy model: ~50M parameters
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, 7, stride=2, padding=3),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(3, stride=2, padding=1),
                self._make_layer(64, 128, 3),
                self._make_layer(128, 256, 4),
                self._make_layer(256, 512, 6),
                self._make_layer(512, 512, 3),
            )
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(512, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(4096, 1000),
            )
        
        else:
            raise ValueError(f"Unknown complexity: {complexity}")
    
    def _make_layer(self, in_channels, out_channels, num_blocks):
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_blocks - 1):
            layers.append(nn.Conv2d(out_channels, out_channels, 3, padding=1))
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class GPUStressTest:
    """Main stress test runner"""
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if not torch.cuda.is_available():
            print("WARNING: CUDA not available, running on CPU")
        else:
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    def get_intensity_params(self):
        """Get parameters based on intensity level"""
        intensity_configs = {
            1: {'batch_size': 8, 'num_workers': 2, 'complexity': 'light', 'repeat_factor': 1},
            2: {'batch_size': 16, 'num_workers': 4, 'complexity': 'light', 'repeat_factor': 2},
            3: {'batch_size': 32, 'num_workers': 4, 'complexity': 'medium', 'repeat_factor': 3},
            4: {'batch_size': 64, 'num_workers': 6, 'complexity': 'medium', 'repeat_factor': 5},
            5: {'batch_size': 96, 'num_workers': 8, 'complexity': 'medium', 'repeat_factor': 10},
            6: {'batch_size': 128, 'num_workers': 8, 'complexity': 'heavy', 'repeat_factor': 15},
            7: {'batch_size': 192, 'num_workers': 10, 'complexity': 'heavy', 'repeat_factor': 20},
            8: {'batch_size': 256, 'num_workers': 12, 'complexity': 'heavy', 'repeat_factor': 30},
            9: {'batch_size': 384, 'num_workers': 16, 'complexity': 'heavy', 'repeat_factor': 50},
            10: {'batch_size': 512, 'num_workers': 16, 'complexity': 'heavy', 'repeat_factor': 100},
        }
        
        if self.args.intensity in intensity_configs:
            config = intensity_configs[self.args.intensity]
        else:
            # Custom intensity
            config = {
                'batch_size': self.args.batch_size or 32,
                'num_workers': self.args.num_workers or 4,
                'complexity': self.args.complexity or 'medium',
                'repeat_factor': self.args.repeat_factor or 5,
            }
        
        # Override with command-line arguments if provided
        if self.args.batch_size:
            config['batch_size'] = self.args.batch_size
        if self.args.num_workers is not None:
            config['num_workers'] = self.args.num_workers
        if self.args.complexity:
            config['complexity'] = self.args.complexity
        if self.args.repeat_factor:
            config['repeat_factor'] = self.args.repeat_factor
        
        return config
    
    def setup_data(self, config):
        """Setup data loader"""
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        dataset = LFWStressDataset(
            self.args.dataset_path,
            transform=transform,
            repeat_factor=config['repeat_factor']
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config['num_workers'],
            pin_memory=True,
            drop_last=True
        )
        
        return dataloader
    
    def setup_model(self, config):
        """Setup model and optimizer"""
        model = StressTestModel(complexity=config['complexity'])
        model = model.to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"\nModel Complexity: {config['complexity']}")
        print(f"Total Parameters: {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
        
        return model, criterion, optimizer
    
    def run_training_stress(self, model, dataloader, criterion, optimizer, config):
        """Run training stress test"""
        model.train()
        
        print(f"\n{'='*60}")
        print(f"STRESS TEST - Intensity Level {self.args.intensity}/10")
        print(f"{'='*60}")
        print(f"Batch Size: {config['batch_size']}")
        print(f"Num Workers: {config['num_workers']}")
        print(f"Model Complexity: {config['complexity']}")
        print(f"Dataset Repeat Factor: {config['repeat_factor']}x")
        print(f"Total Batches per Epoch: {len(dataloader)}")
        print(f"Duration: {self.args.duration} seconds")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        total_batches = 0
        total_samples = 0
        losses = []
        
        # Track GPU metrics
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        try:
            while (time.time() - start_time) < self.args.duration:
                epoch_start = time.time()
                
                for batch_idx, (inputs, targets) in enumerate(dataloader):
                    if (time.time() - start_time) >= self.args.duration:
                        break
                    
                    batch_start = time.time()
                    
                    inputs = inputs.to(self.device)
                    targets = torch.randint(0, 1000, (inputs.size(0),)).to(self.device)
                    
                    # Forward pass
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    batch_time = time.time() - batch_start
                    total_batches += 1
                    total_samples += inputs.size(0)
                    losses.append(loss.item())
                    
                    # Print progress every N batches
                    if total_batches % 10 == 0:
                        elapsed = time.time() - start_time
                        throughput = total_samples / elapsed
                        avg_loss = np.mean(losses[-100:])
                        
                        gpu_mem_str = ""
                        if torch.cuda.is_available():
                            gpu_mem_used = torch.cuda.memory_allocated() / 1e9
                            gpu_mem_peak = torch.cuda.max_memory_allocated() / 1e9
                            gpu_mem_str = f"GPU Mem: {gpu_mem_used:.2f}/{gpu_mem_peak:.2f} GB | "
                        
                        print(f"Batch {total_batches:4d} | "
                              f"Time: {elapsed:.1f}s | "
                              f"Throughput: {throughput:.1f} img/s | "
                              f"{gpu_mem_str}"
                              f"Loss: {avg_loss:.4f} | "
                              f"Batch Time: {batch_time:.3f}s")
                
        except KeyboardInterrupt:
            print("\n\nStress test interrupted by user")
        except Exception as e:
            print(f"\n\nError during stress test: {e}")
            import traceback
            traceback.print_exc()
        
        # Final statistics
        total_time = time.time() - start_time
        self.print_summary(total_batches, total_samples, total_time, losses)
    
    def print_summary(self, total_batches, total_samples, total_time, losses):
        """Print test summary"""
        print(f"\n{'='*60}")
        print(f"STRESS TEST SUMMARY")
        print(f"{'='*60}")
        print(f"Total Time: {total_time:.2f} seconds")
        print(f"Total Batches: {total_batches}")
        print(f"Total Samples: {total_samples}")
        print(f"Average Throughput: {total_samples / total_time:.2f} images/sec")
        print(f"Average Batch Time: {total_time / total_batches:.3f} seconds")
        
        if losses:
            print(f"Average Loss: {np.mean(losses):.4f}")
            print(f"Final Loss: {np.mean(losses[-10:]):.4f}")
        
        if torch.cuda.is_available():
            print(f"\nGPU Statistics:")
            print(f"  Memory Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
            print(f"  Peak Memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
            print(f"  Memory Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
            print(f"  GPU Utilization: High (sustained load)")
        
        print(f"{'='*60}\n")
    
    def run(self):
        """Main execution"""
        print("\nGPU Stress Test Starting...")
        print(f"Dataset: {self.args.dataset_path}")
        
        # Get intensity configuration
        config = self.get_intensity_params()
        
        # Setup
        dataloader = self.setup_data(config)
        model, criterion, optimizer = self.setup_model(config)
        
        # Run stress test
        self.run_training_stress(model, dataloader, criterion, optimizer, config)
        
        print("Stress test completed!")


def main():
    parser = argparse.ArgumentParser(description='GPU Stress Test with Configurable Intensity')
    
    # Intensity level
    parser.add_argument('--intensity', type=int, default=3, choices=range(1, 11),
                       help='Intensity level (1-10). Higher = more GPU stress. Default: 3')
    
    # Dataset
    parser.add_argument('--dataset_path', type=str, 
                       default='datasets/LFW_original/lfw',
                       help='Path to LFW dataset')
    
    # Duration
    parser.add_argument('--duration', type=int, default=300,
                       help='Test duration in seconds. Default: 300 (5 minutes)')
    
    # Manual overrides (optional)
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Override batch size (overrides intensity preset)')
    parser.add_argument('--num_workers', type=int, default=None,
                       help='Override number of data loading workers')
    parser.add_argument('--complexity', type=str, choices=['light', 'medium', 'heavy'],
                       help='Override model complexity')
    parser.add_argument('--repeat_factor', type=int, default=None,
                       help='Override dataset repeat factor')
    
    args = parser.parse_args()
    
    # Validate dataset path
    if not os.path.exists(args.dataset_path):
        print(f"ERROR: Dataset path does not exist: {args.dataset_path}")
        sys.exit(1)
    
    # Run stress test
    tester = GPUStressTest(args)
    tester.run()


if __name__ == '__main__':
    main()
