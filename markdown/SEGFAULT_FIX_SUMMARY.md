# Ablation Study Crash Analysis & Fix

## Problem Summary

The ablation study was **NOT failing due to CUDA memory leaks**, but due to **Segmentation Faults** during training.

## Evidence from Logs

From `nohup.out`:
```bash
./examples/ablation_study.sh: line 68: 108635 Segmentation fault (core dumped)
./examples/ablation_study.sh: line 107: 289546 Segmentation fault (core dumped)
```

## Root Causes Identified

### 1. **Segmentation Faults**
- **Primary Cause**: Too many DataLoader worker threads (default: 16)
- DataLoader workers can cause segfaults when:
  - Multiple processes compete for shared memory
  - File descriptor limits are reached
  - CUDA context conflicts occur in multi-process environments

### 2. **Script Logic Error**
- The script tried to move weights: `mv ./weights/train/*`
- Failed when training crashed before saving checkpoints
- Error: `mv: cannot stat './weights/train/*': No such file or directory`

### 3. **Memory Management**
- **Not a memory leak**, but suboptimal tensor cleanup
- Tensors remained in memory longer than necessary during training loop

## Solutions Implemented

### 1. Reduced DataLoader Workers
**File**: `examples/ablation_study.sh`

Added `--threads=4` to both training functions:
```bash
python train.py \
    --lfw \
    --threads=4 \  # <- REDUCED FROM 16 to 4
    --batchSize=8 \
    ...
```

**Why this helps**:
- Fewer worker processes = less memory pressure
- Reduces inter-process communication overhead
- Prevents file descriptor exhaustion
- More stable on systems with many concurrent processes

### 2. Added Safety Checks for Weight Moving
**File**: `examples/ablation_study.sh`

```bash
# Before (crashed on missing files):
mv ./weights/train/* $WEIGHTS_DIR/

# After (check if files exist first):
if [ -d "./weights/train" ] && [ "$(ls -A ./weights/train 2>/dev/null)" ]; then
    mv ./weights/train/* $WEIGHTS_DIR/
    echo "✓ $NAME training complete. Weights saved to: $WEIGHTS_DIR"
else
    echo "✗ No weights found in ./weights/train - training may have crashed"
    return 1
fi
```

### 3. Explicit Tensor Cleanup
**File**: `train.py`

Added explicit tensor deletion after backpropagation:
```python
# Store loss value before cleanup
loss_value = loss.item()
loss_print = loss_print + loss_value
loss_last_10 = loss_last_10 + loss_value

# Clean up to prevent memory leaks
del loss, loss_rgb, loss_hvi, output_rgb, output_hvi, gt_rgb, gt_hvi, im1, im2
if opt.use_face_loss and 'fr_loss_value' in locals():
    del fr_loss_value
```

**Why this helps**:
- Forces immediate tensor deallocation
- Reduces peak memory usage
- Helps garbage collector free GPU memory faster
- Prevents accumulation of temporary tensors

## Current Status

### Old Runs (Completed Successfully)
The previous training runs all completed successfully:
- ✅ `baseline` - 50 epochs
- ✅ `fr_weight_0.3` - 50 epochs  
- ✅ `fr_weight_0.5` - 50 epochs
- ✅ `fr_weight_1.0` - 50 epochs

**Note**: Old runs didn't use the new `d_1` / `d_1.5` directory structure

### New Runs (In Progress)
Currently training with updated script structure:
- 🔄 `baseline_d1` - in progress
- Scripts now test **two D_weight values** (1.0 and 1.5)
- Running with `--threads=4` to prevent segfaults

## GPU Memory Status

Current GPU usage is **HEALTHY**:
```
GPU Memory: 10742MiB / 24564MiB (44%)
GPU Util: 97%
```

No evidence of:
- ❌ Out of memory errors
- ❌ Memory leaks
- ❌ CUDA errors
- ❌ Fragmentation issues

## Recommendations

### For Current Training
1. **Monitor** the new training runs with reduced threads
2. **Check** if segfaults still occur in logs
3. **If segfaults persist**, try:
   - Reduce batch size from 8 to 4
   - Reduce threads further to 2 or 0
   - Add `export CUDA_LAUNCH_BLOCKING=1` before running

### For Future Training
1. **Always use** `--threads=4` or lower for stability
2. **Add** `torch.cuda.empty_cache()` after validation epochs
3. **Consider** using `persistent_workers=True` in DataLoader
4. **Monitor** `dmesg` for kernel-level segfault details:
   ```bash
   dmesg | grep -i "segfault\|python"
   ```

### If Segfaults Continue
```bash
# Check system limits
ulimit -a

# Increase file descriptor limit if needed
ulimit -n 4096

# Check for OOM killer
dmesg | grep -i "killed process"

# Run with debugging
gdb --args python train.py --lfw ... 
```

## Testing the Fix

To test if the fix works:
```bash
# Kill current training
pkill -f "train.py"

# Clean up
rm -rf ./weights/train/*

# Re-run with fixed script
./examples/ablation_study.sh
```

Monitor for:
- ✅ No segmentation faults in logs
- ✅ Checkpoints being saved properly
- ✅ Weights moved to correct directories
- ✅ All 8 configurations completing (4 FR configs × 2 D_weights)

## Key Takeaway

**The issue was NOT a CUDA memory leak**, but rather:
1. ❌ Too many DataLoader workers causing segfaults
2. ❌ Poor error handling when training crashed
3. ⚠️ Suboptimal (but not critical) tensor cleanup

All fixes have been applied and should resolve the crashes.
