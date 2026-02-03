"""
Extract Evaluation Data from face_eval.log

Parses face_eval.log and exports clean data to JSON format for visualization.

Usage:
    python extract_evaluation_data.py --eval_log=./face_eval.log --output=./data/evaluation_data.json
"""

import os
import re
import argparse
import json
from collections import defaultdict


def parse_face_eval_log(filepath):
    """Parse face verification results from eval log file

    Expected log format:
        [Mon Feb  2 09:47:27 AM UTC 2026] Evaluating: baseline on easy
        ...
        Equal Error Rate (EER):
            Enhanced:   X.XX%
        True Accept Rate @ FAR=0.1%:
            Enhanced:   XX.XX%
        True Accept Rate @ FAR=1%:
            Enhanced:   XX.XX%
        Genuine Pair Scores:
            Enhanced:   X.XXXX
        Average PSNR: X.XX dB
        Average SSIM: X.XXXX
    """
    results = defaultdict(lambda: defaultdict(dict))

    if not os.path.exists(filepath):
        print(f"Error: Log file not found: {filepath}")
        return None

    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    # Split into evaluation sections by looking for the evaluation header
    # Pattern: [timestamp] Evaluating: <model> on <difficulty>
    eval_pattern = r'\[.*?\]\s+Evaluating:\s+(\S+)\s+on\s+(\w+)'

    # Find all matches with their positions
    matches = list(re.finditer(eval_pattern, content))

    for i, match in enumerate(matches):
        model = match.group(1)
        difficulty = match.group(2)

        # Get the section content (from this match to next match or end)
        start_pos = match.end()
        if i + 1 < len(matches):
            end_pos = matches[i + 1].start()
        else:
            end_pos = len(content)

        section_content = content[start_pos:end_pos]

        # Parse EER
        match_eer = re.search(r'Equal Error Rate.*?Enhanced:\s+([\d.]+)%', section_content, re.DOTALL)
        if match_eer:
            results[model][difficulty]['eer'] = float(match_eer.group(1))

        # Parse TAR @ FAR=0.1%
        match_tar001 = re.search(r'TAR.*?FAR=0\.1%.*?Enhanced:\s+([\d.]+)%', section_content, re.DOTALL)
        if match_tar001:
            results[model][difficulty]['tar_001'] = float(match_tar001.group(1))

        # Parse TAR @ FAR=1%
        match_tar01 = re.search(r'TAR.*?FAR=1%.*?Enhanced:\s+([\d.]+)%', section_content, re.DOTALL)
        if match_tar01:
            results[model][difficulty]['tar_01'] = float(match_tar01.group(1))

        # Parse genuine similarity
        match_sim = re.search(r'Genuine Pair Scores.*?Enhanced:\s+([\d.]+)', section_content, re.DOTALL)
        if match_sim:
            results[model][difficulty]['genuine_similarity'] = float(match_sim.group(1))

        # Parse PSNR
        match_psnr = re.search(r'PSNR:\s+([\d.]+)', section_content)
        if match_psnr:
            results[model][difficulty]['psnr'] = float(match_psnr.group(1))

        # Parse SSIM
        match_ssim = re.search(r'SSIM:\s+([\d.]+)', section_content)
        if match_ssim:
            results[model][difficulty]['ssim'] = float(match_ssim.group(1))

        # Also parse low-light metrics for comparison
        match_ll_eer = re.search(r'Equal Error Rate.*?Low-light:\s+([\d.]+)%', section_content, re.DOTALL)
        if match_ll_eer:
            results[model][difficulty]['low_light_eer'] = float(match_ll_eer.group(1))

    # Convert defaultdict to regular dict
    return {k: dict(v) for k, v in results.items()}


def main():
    parser = argparse.ArgumentParser(description='Extract evaluation data from log file')
    parser.add_argument('--eval_log', type=str, default='./face_eval.log',
                       help='Path to face evaluation log file')
    parser.add_argument('--output', type=str, default='./data/evaluation_data.json',
                       help='Output JSON file path')

    args = parser.parse_args()

    print("=" * 70)
    print("Extracting Evaluation Data")
    print("=" * 70)
    print(f"Input:  {args.eval_log}")
    print(f"Output: {args.output}")
    print()

    # Create output directory
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Parse log file
    data = parse_face_eval_log(args.eval_log)

    if not data:
        print("No evaluation data found!")
        return 1

    print(f"Found evaluation data for {len(data)} model(s):")
    for model_name in sorted(data.keys()):
        difficulties = list(data[model_name].keys())
        print(f"  - {model_name}: {difficulties}")
        for diff in difficulties:
            metrics = data[model_name][diff]
            if 'eer' in metrics:
                print(f"    {diff}: EER={metrics['eer']:.2f}%, PSNR={metrics.get('psnr', 'N/A')}, SSIM={metrics.get('ssim', 'N/A')}")
    print()

    # Save to JSON
    with open(args.output, 'w') as f:
        json.dump(data, f, indent=2)

    print("=" * 70)
    print(f"Evaluation data saved to: {args.output}")
    print("=" * 70)

    return 0


if __name__ == '__main__':
    exit(main())
