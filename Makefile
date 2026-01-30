baseline:
	TQDM_DISABLE=1 python train.py --lfw --data_train_lfw=datasets/LFW_multilevel/train_mixed \
		--data_val_lfw=datasets/LFW_multilevel/val_mixed \
		--data_valgt_lfw=datasets/LFW_multilevel/val_mixed/high --D_weight=1.5 --FR_weight=0.0 \
		--batchSize=8 --nEpochs=200

face_loss3:
	TQDM_DISABLE=1 python train.py --lfw --data_train_lfw=datasets/LFW_multilevel/train_mixed \
		--data_val_lfw=datasets/LFW_multilevel/val_mixed \
		--data_valgt_lfw=datasets/LFW_multilevel/val_mixed/high --D_weight=1.5 \
		--batchSize=8 --nEpochs=200 --use_face_loss --FR_weight=0.3

face_loss5:
	TQDM_DISABLE=1 python train.py --lfw --data_train_lfw=datasets/LFW_multilevel/train_mixed \
		--data_val_lfw=datasets/LFW_multilevel/val_mixed \
		--data_valgt_lfw=datasets/LFW_multilevel/val_mixed/high --D_weight=1.5 \
		--batchSize=8 --nEpochs=200 --use_face_loss --FR_weight=0.5