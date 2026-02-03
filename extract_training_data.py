"""
Extract Training Data from train.log

Parses train.log and exports clean data to JSON format for visualization.

Usage:
    python extract_training_data.py --train_log=./train.log --output=./data/training_data.json
"""

import os
import re
import argparse
import json
from collections import defaultdict


def parse_train_log(filepath):
    """Parse training log file - extracts LAST complete training run for each model"""
    if not os.path.exists(filepath):
        print(f"Error: Log file not found: {filepath}")
        return None

    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    # Track ALL training runs for each model, then select the longest one
    all_runs = {
        'baseline': [],
        'face_loss3': [],
        'face_loss5': [],
    }

    current_model = None
    current_run_data = None

    for line in lines:
        # Detect model switches
        if 'Starting training:' in line:
            match = re.search(r'Starting training:\s+(\S+)', line)
            if match:
                model_name_raw = match.group(1)
                if 'baseline' in model_name_raw:
                    new_model = 'baseline'
                elif 'face_loss3' in model_name_raw or '0.3' in model_name_raw:
                    new_model = 'face_loss3'
                elif 'face_loss5' in model_name_raw or '0.5' in model_name_raw:
                    new_model = 'face_loss5'
                else:
                    new_model = None

                # Save previous run if exists
                if current_model and current_run_data and current_run_data['epochs']:
                    all_runs[current_model].append(current_run_data)

                # Start new run
                current_model = new_model
                current_run_data = {'epochs': [], 'losses': [], 'psnr': [], 'ssim': [], 'lr': []}
            continue

        if current_model is None or current_run_data is None:
            continue

        # Parse epoch loss
        match = re.search(r'Epoch\[(\d+)\]:\s+Loss:\s+([\d.]+)\s+\|\|.*?lr=([\d.e+-]+)', line)
        if match:
            epoch = int(match.group(1))
            loss = float(match.group(2))
            lr_str = match.group(3).rstrip('.')
            try:
                lr = float(lr_str)
            except ValueError:
                lr = 0.0

            # Check for restart within same model (epoch decreased)
            if current_run_data['epochs'] and epoch <= current_run_data['epochs'][-1]:
                # Save previous run and start new one
                all_runs[current_model].append(current_run_data)
                current_run_data = {'epochs': [], 'losses': [], 'psnr': [], 'ssim': [], 'lr': []}

            current_run_data['epochs'].append(epoch)
            current_run_data['losses'].append(loss)
            current_run_data['lr'].append(lr)

        # Parse PSNR
        match = re.search(r'Avg\.PSNR:\s+([\d.]+)', line)
        if match:
            psnr = float(match.group(1))
            current_run_data['psnr'].append(psnr)

        # Parse SSIM
        match = re.search(r'Avg\.SSIM:\s+([\d.]+)', line)
        if match:
            ssim = float(match.group(1))
            current_run_data['ssim'].append(ssim)

    # Save last run
    if current_model and current_run_data and current_run_data['epochs']:
        all_runs[current_model].append(current_run_data)

    # For each model, combine all training runs to get complete sequence
    training_data = {}
    for model, runs in all_runs.items():
        if not runs:
            training_data[model] = {'epochs': [], 'losses': [], 'psnr': [], 'ssim': [], 'lr': []}
            continue

        # Combine all runs, merging epochs that form a sequence
        combined = {'epochs': [], 'losses': [], 'psnr': [], 'ssim': [], 'lr': []}
        seen_epochs = set()

        for run in runs:
            for i in range(len(run['epochs'])):
                epoch = run['epochs'][i]
                if epoch not in seen_epochs:
                    seen_epochs.add(epoch)
                    combined['epochs'].append(epoch)
                    combined['losses'].append(run['losses'][i])
                    combined['lr'].append(run['lr'][i])

            # Merge PSNR/SSIM (may have duplicates, that's ok)
            combined['psnr'].extend(run.get('psnr', []))
            combined['ssim'].extend(run.get('ssim', []))

        # Sort by epoch
        sorted_indices = sorted(range(len(combined['epochs'])), key=lambda i: combined['epochs'][i])
        combined['epochs'] = [combined['epochs'][i] for i in sorted_indices]
        combined['losses'] = [combined['losses'][i] for i in sorted_indices]
        combined['lr'] = [combined['lr'][i] for i in sorted_indices]

        training_data[model] = combined

    return training_data


def main():
    parser = argparse.ArgumentParser(description='Extract training data from log file')
    parser.add_argument('--train_log', type=str, default='./train.log',
                       help='Path to training log file')
    parser.add_argument('--output', type=str, default='./data/training_data.json',
                       help='Output JSON file path')

    args = parser.parse_args()

    print("=" * 70)
    print("Extracting Training Data")
    print("=" * 70)
    print(f"Input:  {args.train_log}")
    print(f"Output: {args.output}")
    print()

    # Create output directory
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Parse log file
    data = parse_train_log(args.train_log)

    if not data:
        print("No training data found!")
        return 1

    print(f"Found training data for {len(data)} model(s):")
    for model_name, model_data in data.items():
        epochs = len(model_data['epochs'])
        print(f"  - {model_name}: {epochs} epochs")
        if model_data['losses']:
            print(f"    Loss range: {min(model_data['losses']):.4f} - {max(model_data['losses']):.4f}")
        print(f"    PSNR samples: {len(model_data.get('psnr', []))}, SSIM samples: {len(model_data.get('ssim', []))}")
        if model_data['epochs']:
            print(f"    Epoch range: {min(model_data['epochs'])} - {max(model_data['epochs'])}")
    print()

    # Save to JSON
    with open(args.output, 'w') as f:
        json.dump(data, f, indent=2)

    print("=" * 70)
    print(f"Training data saved to: {args.output}")
    print("=" * 70)

    return 0


if __name__ == '__main__':
    exit(main())
