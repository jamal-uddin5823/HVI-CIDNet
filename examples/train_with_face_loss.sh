#!/bin/bash
# Train with Face Recognition Perceptual Loss
# This is your thesis contribution!

echo "========================================================================"
echo "Training with Face Recognition Perceptual Loss"
echo "========================================================================"

# Check if AdaFace weights exist
ADAFACE_WEIGHTS="./pretrained/adaface/adaface_ir50_webface4m.ckpt"

if [ -f "$ADAFACE_WEIGHTS" ]; then
    echo "✓ Using pre-trained AdaFace weights: $ADAFACE_WEIGHTS"
    WEIGHTS_ARG="--FR_model_path=$ADAFACE_WEIGHTS"
else
    echo "⚠ AdaFace weights not found at: $ADAFACE_WEIGHTS"
    echo "  Using ResNet50 fallback (less accurate)"
    echo "  Download weights from: https://github.com/mk-minchul/AdaFace"
    WEIGHTS_ARG=""
fi

echo ""
echo "Starting training..."
echo ""

python train.py \
    --lfw=True \
    --data_train_lfw=./datasets/LFW_lowlight/train \
    --data_val_lfw=./datasets/LFW_lowlight/val \
    --data_valgt_lfw=./datasets/LFW_lowlight/val/high \
    --batchSize=8 \
    --cropSize=256 \
    --nEpochs=100 \
    --lr=0.0001 \
    --L1_weight=1.0 \
    --D_weight=0.5 \
    --E_weight=50.0 \
    --P_weight=0.01 \
    --use_face_loss=True \
    --FR_weight=0.5 \
    --FR_model_arch=ir_50 \
    --FR_feature_distance=mse \
    --threads=8 \
    --snapshots=10 \
    $WEIGHTS_ARG

echo ""
echo "========================================================================"
echo "Training complete!"
echo "Checkpoints saved to: ./weights/train/"
echo "Results saved to: ./results/"
echo "========================================================================"
