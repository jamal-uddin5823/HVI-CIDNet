"""
Extract and validate data for thesis visualizations
Run this FIRST to ensure data is correctly loaded
"""
import json
import os
import numpy as np
from pathlib import Path

def validate_and_extract():
    """Extract all data with validation checks"""

    # ==============================================================
    # 1. TRAINING DATA (epochs 1-40 only)
    # ==============================================================
    print("=" * 70)
    print("EXTRACTING TRAINING DATA")
    print("=" * 70)

    with open('data/training_data.json', 'r') as f:
        training_data = json.load(f)

    # Extract first 40 epochs for all models
    extracted_training = {}
    for model_name in ['baseline', 'face_loss3', 'face_loss5']:
        model_data = training_data[model_name]

        # Find index for epoch 40 (inclusive)
        epochs = model_data['epochs']
        idx_40 = epochs.index(40) + 1 if 40 in epochs else len(epochs)

        extracted_training[model_name] = {
            'epochs': epochs[:idx_40],
            'losses': model_data['losses'][:idx_40],
            'psnr': model_data.get('psnr', []),
            'ssim': model_data.get('ssim', []),
            'lr': model_data['lr'][:idx_40]
        }

        print(f"\n{model_name}:")
        print(f"  Epochs: {len(extracted_training[model_name]['epochs'])} (1-{extracted_training[model_name]['epochs'][-1]})")
        print(f"  Loss range: {min(extracted_training[model_name]['losses']):.4f} - {max(extracted_training[model_name]['losses']):.4f}")
        print(f"  Validation points: {len(extracted_training[model_name]['psnr'])}")

        # VALIDATION CHECK
        assert len(extracted_training[model_name]['epochs']) > 0, f"{model_name}: No epochs found!"
        assert len(extracted_training[model_name]['losses']) == len(extracted_training[model_name]['epochs']), \
            f"{model_name}: Epoch/loss length mismatch!"

    # ==============================================================
    # 2. EVALUATION DATA
    # ==============================================================
    print("\n" + "=" * 70)
    print("EXTRACTING EVALUATION DATA")
    print("=" * 70)

    with open('data/evaluation_data.json', 'r') as f:
        eval_data = json.load(f)

    # Validate structure
    models = ['baseline', 'face_loss3', 'face_loss5']
    difficulties = ['easy', 'medium', 'hard', 'mixed']
    metrics = ['eer', 'tar_001', 'tar_01', 'genuine_similarity', 'psnr', 'ssim', 'low_light_eer']

    for model in models:
        for diff in difficulties:
            assert model in eval_data, f"Missing model: {model}"
            assert diff in eval_data[model], f"Missing difficulty: {diff} in {model}"

            for metric in metrics:
                assert metric in eval_data[model][diff], f"Missing metric: {metric} in {model}/{diff}"

            print(f"{model}/{diff}: EER={eval_data[model][diff]['eer']:.2f}%, TAR@0.1%={eval_data[model][diff]['tar_001']:.1f}%")

    # ==============================================================
    # 3. ROC DATA (from verification_scores.json files)
    # ==============================================================
    print("\n" + "=" * 70)
    print("EXTRACTING ROC DATA")
    print("=" * 70)

    roc_data = {}
    for model in models:
        roc_data[model] = {}
        for diff in difficulties:
            json_path = f'results/multilevel_evaluations/{model}/{diff}/verification_scores.json'

            if not os.path.exists(json_path):
                print(f"WARNING: {json_path} not found - skipping")
                continue

            with open(json_path, 'r') as f:
                scores_data = json.load(f)

            # Extract ROC curve data
            roc_data[model][diff] = {
                'genuine_scores_enhanced': scores_data['genuine_scores_enhanced'],
                'impostor_scores_enhanced': scores_data['impostor_scores_enhanced'],
                'tpr': scores_data['roc_data']['enhanced']['tpr'],
                'fpr': scores_data['roc_data']['enhanced']['fpr'],
                'thresholds': scores_data['roc_data']['enhanced']['thresholds']
            }

            print(f"{model}/{diff}:")
            print(f"  Genuine scores: {len(roc_data[model][diff]['genuine_scores_enhanced'])} pairs")
            print(f"  Impostor scores: {len(roc_data[model][diff]['impostor_scores_enhanced'])} pairs")
            print(f"  ROC points: {len(roc_data[model][diff]['tpr'])}")

            # VALIDATION CHECK
            assert len(roc_data[model][diff]['tpr']) == len(roc_data[model][diff]['fpr']), \
                f"{model}/{diff}: TPR/FPR length mismatch!"

    # ==============================================================
    # 4. SAVE EXTRACTED DATA
    # ==============================================================
    output = {
        'training': extracted_training,
        'evaluation': eval_data,
        'roc': roc_data
    }

    with open('thesis_data_extracted.json', 'w') as f:
        json.dump(output, f, indent=2)

    print("\n" + "=" * 70)
    print("✓ DATA EXTRACTION COMPLETE")
    print("=" * 70)
    print(f"Saved to: thesis_data_extracted.json")
    print(f"File size: {os.path.getsize('thesis_data_extracted.json') / 1024:.1f} KB")

    return output

if __name__ == '__main__':
    data = validate_and_extract()
    print("\n✓ All validations passed - ready for plotting!")
