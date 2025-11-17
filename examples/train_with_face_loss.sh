#!/bin/bash
# Train with Face Recognition Perceptual Loss
# This is your thesis contribution!

echo "========================================================================"
echo "Training with Face Recognition Perceptual Loss (Fine-tuning)"
echo "========================================================================"

# Check if AdaFace weights exist
ADAFACE_WEIGHTS="./weghts/adaface/adaface_ir50_webface4m.ckpt"
CIDNET_WEIGHTS="./weights/LOLv2_real/best_PSNR.pth"

if [ -f "$ADAFACE_WEIGHTS" ]; then
    echo "✓ Using pre-trained AdaFace weights: $ADAFACE_WEIGHTS"
    ADAFACE_ARG="--FR_model_path=$ADAFACE_WEIGHTS"
else
    echo "⚠ AdaFace weights not found at: $ADAFACE_WEIGHTS"
    echo "  Using randomly initialized weights (NOT recommended)"
    echo "  Download weights from: https://github.com/mk-minchul/AdaFace/releases"
    ADAFACE_ARG=""
fi

if [ -f "$CIDNET_WEIGHTS" ]; then
    echo "✓ Fine-tuning from: $CIDNET_WEIGHTS"
    CIDNET_ARG="--pretrained_model=$CIDNET_WEIGHTS"
    LR="0.00001"  # Lower learning rate for fine-tuning
    EPOCHS="50"   # Fewer epochs needed for fine-tuning
else
    echo "⚠ CIDNet pretrained weights not found at: $CIDNET_WEIGHTS"
    echo "  Training from scratch"
    CIDNET_ARG=""
    LR="0.0001"   # Higher learning rate for training from scratch
    EPOCHS="100"  # More epochs needed for training from scratch
fi

echo ""
echo "Starting training..."
echo ""

python train.py \
    --lfw \
    --data_train_lfw=./datasets/LFW_lowlight/train \
    --data_val_lfw=./datasets/LFW_lowlight/val \
    --data_valgt_lfw=./datasets/LFW_lowlight/val/high \
    $CIDNET_ARG \
    --batchSize=8 \
    --cropSize=256 \
    --nEpochs=$EPOCHS \
    --lr=$LR \
    --L1_weight=1.0 \
    --D_weight=0.5 \
    --E_weight=50.0 \
    --P_weight=0.01 \
    --use_face_loss \
    --FR_weight=0.5 \
    --FR_model_arch=ir_50 \
    --FR_feature_distance=mse \
    --threads=8 \
    --snapshots=10 \
    $ADAFACE_ARG

echo ""
echo "========================================================================"
echo "Training complete!"
echo "Checkpoints saved to: ./weights/train/"
echo "Results saved to: ./results/"
echo "========================================================================"
