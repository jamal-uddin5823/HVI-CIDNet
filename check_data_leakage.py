"""
Check for Data Leakage in Your Experiment

This script helps verify that your training and evaluation sets are properly separated.
"""

import os
from pathlib import Path
from collections import defaultdict

def check_train_test_overlap(train_dir, test_dir):
    """
    Check if training and test sets have overlapping images.
    """
    print("="*70)
    print("DATA LEAKAGE CHECK")
    print("="*70)
    
    # Get all filenames (not full paths)
    train_files = set()
    test_files = set()
    
    print(f"\nScanning training directory: {train_dir}")
    for root, dirs, files in os.walk(train_dir):
        for f in files:
            if f.endswith(('.jpg', '.png', '.jpeg')):
                # Get base filename without _lowlight suffix
                base = f.replace('_lowlight', '').replace('_rf01', '').replace('_rf03', '')
                base = base.replace('_rf05', '').replace('_rf10', '').replace('_rf30', '')
                train_files.add(base)
    
    print(f"Scanning test directory: {test_dir}")
    for root, dirs, files in os.walk(test_dir):
        for f in files:
            if f.endswith(('.jpg', '.png', '.jpeg')):
                base = f.replace('_lowlight', '').replace('_rf01', '').replace('_rf03', '')
                base = base.replace('_rf05', '').replace('_rf10', '').replace('_rf30', '')
                test_files.add(base)
    
    print(f"\nTraining images: {len(train_files)}")
    print(f"Test images: {len(test_files)}")
    
    # Check overlap
    overlap = train_files & test_files
    
    print(f"\n{'='*70}")
    if len(overlap) > 0:
        print(f"❌ DATA LEAKAGE DETECTED!")
        print(f"{'='*70}")
        print(f"Number of overlapping images: {len(overlap)}")
        print(f"Overlap percentage: {len(overlap)/len(test_files)*100:.1f}% of test set")
        print(f"\nThis means your model has seen these test images during training!")
        print(f"Results are INVALID - you're testing on training data.")
        print(f"\nFirst 10 overlapping files:")
        for i, f in enumerate(list(overlap)[:10]):
            print(f"  {i+1}. {f}")
    else:
        print(f"✓ NO DATA LEAKAGE")
        print(f"{'='*70}")
        print(f"Training and test sets are properly separated.")
        print(f"Results are valid.")
    
    return len(overlap) == 0


def verify_lfw_split(lfw_root):
    """
    Verify LFW dataset is split correctly according to standard protocol.
    
    LFW standard: 10-fold cross-validation
    - Each fold: train on 9 folds, test on 1 fold
    - Test pairs should use images from the test fold only
    """
    print("\n" + "="*70)
    print("LFW DATASET SPLIT VERIFICATION")
    print("="*70)
    
    # Check if using standard LFW pairs
    pairs_file = Path(lfw_root) / "pairs.txt"
    pairsDevTrain = Path(lfw_root) / "pairsDevTrain.txt"
    pairsDevTest = Path(lfw_root) / "pairsDevTest.txt"
    
    if pairs_file.exists():
        print(f"\n✓ Found standard pairs.txt file")
        print(f"  This contains the test pairs for evaluation")
    
    if pairsDevTrain.exists() and pairsDevTest.exists():
        print(f"✓ Found development train/test split files")
        print(f"  Using these for development is correct")
    
    # Parse pairs to check identities
    print(f"\nRECOMMENDATION:")
    print(f"For LFW, you should:")
    print(f"1. Train on some identities (e.g., first 4000 images)")
    print(f"2. Test on COMPLETELY DIFFERENT identities")
    print(f"3. OR use the 10-fold protocol:")
    print(f"   - 9 folds for training")
    print(f"   - 1 fold for testing")
    print(f"   - Repeat 10 times, average results")


def check_evaluation_methodology():
    """
    Print questions to help identify evaluation issues.
    """
    print("\n" + "="*70)
    print("EVALUATION METHODOLOGY CHECKLIST")
    print("="*70)
    
    questions = [
        ("1. Are you synthesizing low-light AFTER splitting train/test?",
         "✓ CORRECT: Split first, then synthesize separately",
         "✗ WRONG: Synthesize first, then split (can leak info)"),
        
        ("2. Are test images completely unseen during training?",
         "✓ CORRECT: Model never trained on test image identities",
         "✗ WRONG: Test images appeared in training set"),
        
        ("3. Are you evaluating on a held-out test set?",
         "✓ CORRECT: Separate test set never used in training",
         "✗ WRONG: Evaluating on training data"),
        
        ("4. Are you using the same images at different darkness levels?",
         "⚠ CAREFUL: This is OK but means model sees same faces",
         "✓ BETTER: Use completely different images per difficulty"),
        
        ("5. Did you use early stopping on validation set?",
         "✓ CORRECT: Separate validation set for early stopping",
         "✗ WRONG: Using test set for early stopping (leakage!)"),
    ]
    
    for question, correct, wrong in questions:
        print(f"\n{question}")
        print(f"  {correct}")
        print(f"  {wrong}")


def diagnose_high_baseline_performance():
    """
    Help diagnose why baseline performs so well.
    """
    print("\n" + "="*70)
    print("WHY IS BASELINE PERFORMING SO WELL?")
    print("="*70)
    
    print("""
Possible Reasons (in order of likelihood):

1. ❌ DATA LEAKAGE (Most Common)
   - Training and test sets overlap
   - Model has memorized the test images
   - FIX: Ensure completely separate train/test splits
   
2. ✓ STRONG BASELINE (Good Thing!)
   - CIDNet is actually a good architecture
   - Low-light enhancement IS working well
   - This is not a bug - it shows the task is solvable
   
3. ⚠ TEST SET TOO EASY
   - LFW "easy" pairs are frontal, high-quality
   - Even with RF=0.01, still enough structure
   - FIX: Test on harder pairs or different dataset
   
4. ⚠ GAMMA CORRECTION HELPS
   - sRGB gamma makes dark images more visible
   - This is CORRECT behavior, not cheating
   - Real cameras do this too
   
5. ⚠ UNREALISTIC REDUCTION FACTOR
   - RF=0.01 is extremely dark
   - May be beyond realistic camera capability
   - FIX: Use RF=0.05-0.1 for realistic scenarios
   
6. ❌ EVALUATION ON TRAINING DATA
   - Accidentally evaluating on training images
   - This is overfitting, not generalization
   - FIX: Triple-check your evaluation code

WHAT TO DO:
1. Run check_train_test_overlap() first
2. Verify you're using held-out test set
3. If no leakage: baseline IS just working well!
4. Focus on showing your method works BETTER
""")


# ============================================================================
# MAIN DIAGNOSTIC FUNCTION
# ============================================================================

def run_full_diagnostic(train_dir, test_dir, lfw_root=None):
    """
    Run complete diagnostic check.
    """
    print("\n" + "#"*70)
    print("#" + " "*68 + "#")
    print("#" + " "*20 + "DATA LEAKAGE DIAGNOSTIC" + " "*25 + "#")
    print("#" + " "*68 + "#")
    print("#"*70)
    
    # Check 1: Train/Test Overlap
    is_valid = check_train_test_overlap(train_dir, test_dir)
    
    # Check 2: LFW Split (if applicable)
    if lfw_root:
        verify_lfw_split(lfw_root)
    
    # Check 3: Methodology
    check_evaluation_methodology()
    
    # Check 4: Diagnosis
    diagnose_high_baseline_performance()
    
    # Final verdict
    print("\n" + "="*70)
    print("FINAL VERDICT")
    print("="*70)
    
    if not is_valid:
        print("""
❌ DATA LEAKAGE DETECTED!

Your high baseline performance is likely due to the model
seeing test images during training. This invalidates your results.

ACTION REQUIRED:
1. Re-split your dataset ensuring NO overlap
2. Retrain all models from scratch
3. Re-evaluate on the new test set
""")
    else:
        print("""
✓ NO DATA LEAKAGE DETECTED

Your baseline performing well might be legitimate!

POSSIBLE SCENARIOS:
1. CIDNet is actually a strong baseline (good!)
2. Task is easier than expected (need harder test)
3. Your method still provides improvement (show it!)

NEXT STEPS:
1. Verify your evaluation is on truly held-out data
2. Test on harder scenarios (pose variation, occlusion)
3. Show your method provides consistent improvements
4. Quantify the improvement (statistical significance)
""")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("""
USAGE INSTRUCTIONS:
==================

# Check your current setup:
run_full_diagnostic(
    train_dir='path/to/training/images',
    test_dir='path/to/test/images',
    lfw_root='path/to/lfw'  # Optional
)

# Quick check for overlap only:
check_train_test_overlap(
    train_dir='path/to/training/images',
    test_dir='path/to/test/images'
)

IMPORTANT:
- train_dir should contain your TRAINING images
- test_dir should contain your TEST/EVALUATION images
- These should be COMPLETELY DIFFERENT images
- NO image should appear in both directories
""")
    run_full_diagnostic(
        train_dir='datasets/LFW_lowlight/train',
        test_dir='datasets/LFW_lowlight/test',
        lfw_root='datasets/LFW_original'
    )

    check_train_test_overlap(
        train_dir='datasets/LFW_lowlight/train',
        test_dir='datasets/LFW_lowlight/test'
    )
    # Uncomment and modify these paths to run:
    # run_full_diagnostic(
    #     train_dir='/path/to/your/train',
    #     test_dir='/path/to/your/test',
    #     lfw_root='/path/to/lfw'
    # )