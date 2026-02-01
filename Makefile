.PHONY: baseline face_loss3 face_loss5 eval_all eval_baseline eval_face_loss3 eval_face_loss5 visualize_all

baseline:
	TQDM_DISABLE=1 python train.py --lfw --data_train_lfw=datasets/LFW_multilevel/train_mixed \
		--data_val_lfw=datasets/LFW_multilevel/val_mixed \
		--data_valgt_lfw=datasets/LFW_multilevel/val_mixed/high --D_weight=1.5 --FR_weight=0.0 \
		--batchSize=8 --nEpochs=70
	mkdir -p weights/multilevel/baseline
	mv -f weights/train/* weights/multilevel/baseline/ 2>/dev/null || true

face_loss3:
	TQDM_DISABLE=1 python train.py --lfw --data_train_lfw=datasets/LFW_multilevel/train_mixed \
		--data_val_lfw=datasets/LFW_multilevel/val_mixed \
		--data_valgt_lfw=datasets/LFW_multilevel/val_mixed/high --D_weight=1.5 \
		--batchSize=8 --nEpochs=70 --use_face_loss --FR_weight=0.3
	mkdir -p weights/multilevel/face_loss3
	mv -f weights/train/* weights/multilevel/face_loss3/ 2>/dev/null || true

face_loss5:
	TQDM_DISABLE=1 python train.py --lfw --data_train_lfw=datasets/LFW_multilevel/train_mixed \
		--data_val_lfw=datasets/LFW_multilevel/val_mixed \
		--data_valgt_lfw=datasets/LFW_multilevel/val_mixed/high --D_weight=1.5 \
		--batchSize=8 --nEpochs=70 --use_face_loss --FR_weight=0.5
	mkdir -p weights/multilevel/face_loss5
	mv -f weights/train/* weights/multilevel/face_loss5/ 2>/dev/null || true

# Evaluation targets
eval_baseline: baseline
	python eval_face_verification.py --model=./weights/multilevel/baseline/epoch_70.pth \
		--test_dir=./datasets/LFW_multilevel/test_mixed \
		--pairs_file=./datasets/LFW_multilevel/test_mixed/pairs.txt \
		--output_dir=results/multilevel_evaluations/baseline
	mkdir -p results/multilevel_evaluations/baseline

eval_face_loss3: face_loss3
	python eval_face_verification.py --model=./weights/multilevel/face_loss3/epoch_70.pth \
		--test_dir=./datasets/LFW_multilevel/test_mixed \
		--pairs_file=./datasets/LFW_multilevel/test_mixed/pairs.txt \
		--output_dir=results/multilevel_evaluations/face_loss3
	mkdir -p results/multilevel_evaluations/face_loss3

eval_face_loss5: face_loss5
	python eval_face_verification.py --model=./weights/multilevel/face_loss5/epoch_70.pth \
		--test_dir=./datasets/LFW_multilevel/test_mixed \
		--pairs_file=./datasets/LFW_multilevel/test_mixed/pairs.txt \
		--output_dir=results/multilevel_evaluations/face_loss5
	mkdir -p results/multilevel_evaluations/face_loss5

eval_all: eval_baseline eval_face_loss3 eval_face_loss5

# Visualization targets
visualize_all: eval_all
	mkdir -p figures/thesis
	python plot_thesis_summary.py --results_dir=./results/multilevel_evaluations --output_dir=figures/thesis
	python plot_training_curves.py --results_dir=./results/training --output_dir=figures
	python plot_verification_analysis.py --results_dir=./results/multilevel_evaluations --output_dir=figures

# Quick test (1 epoch)
test_baseline:
	TQDM_DISABLE=1 python train.py --lfw --data_train_lfw=datasets/LFW_multilevel/train_mixed \
		--data_val_lfw=datasets/LFW_multilevel/val_mixed \
		--data_valgt_lfw=datasets/LFW_multilevel/val_mixed/high --D_weight=1.5 --FR_weight=0.0 \
		--batchSize=8 --nEpochs=1 --snapshots=1