"""
PATCH FILE FOR train.py - Shows Key Modifications

This file demonstrates the changes needed to integrate:
1. Hard Negative Mining
2. Identity-Balanced Sampling
3. Improved impostor sampling strategy

Apply these changes to train.py to enable the new features.
"""

# ============================================================================
# MODIFICATION 1: Import statements (add after line 18)
# ============================================================================
from loss.discriminative_face_loss import DiscriminativeMultiLevelFaceLoss
from data.hard_negative_sampler import HardNegativeSampler  # NEW
from data.identity_balanced_sampler import IdentityBalancedSampler  # NEW
import gc

# ============================================================================
# MODIFICATION 2: Initialize hard negative sampler (add to init_loss function around line 345)
# ============================================================================
def init_loss():
    L1_weight   = opt.L1_weight
    D_weight    = opt.D_weight
    E_weight    = opt.E_weight
    P_weight    = 1.0

    L1_loss= L1Loss(loss_weight=L1_weight, reduction='mean').cuda()
    D_loss = SSIM(weight=D_weight).cuda()
    E_loss = EdgeLoss(loss_weight=E_weight).cuda()
    P_loss = PerceptualLoss({'conv1_2': 1, 'conv2_2': 1,'conv3_4': 1,'conv4_4': 1}, perceptual_weight = P_weight ,criterion='mse').cuda()

    # Face Recognition Perceptual Loss (optional, for face-specific enhancement)
    FR_loss = None
    hard_neg_sampler = None  # NEW

    if opt.use_face_loss:
        print("===> Initializing Discriminative Multi-Level Face Loss")
        FR_loss = DiscriminativeMultiLevelFaceLoss(
            recognizer_path=opt.FR_model_path,
            architecture=opt.FR_model_arch,
            feature_layers=['layer2', 'layer3', 'layer4', 'fc'],
            layer_weights=[0.2, 0.4, 0.8, 1.0],
            use_contrastive=True,
            contrastive_margin=opt.contrastive_margin,
            contrastive_weight=opt.contrastive_weight,
            use_triplet=True,
            triplet_margin=opt.triplet_margin,
            triplet_weight=opt.triplet_weight,
            temperature=opt.face_temperature
        ).cuda()
        print(f"===> Discriminative Multi-Level Face Loss enabled with weight={opt.FR_weight}")

        # NEW: Initialize hard negative sampler
        if opt.use_hard_negatives:
            print("===> Initializing Hard Negative Sampler")
            hard_neg_sampler = HardNegativeSampler(
                face_recognizer=FR_loss.recognizer,
                memory_size=opt.hard_neg_memory_size,
                topk_hard=opt.hard_neg_topk,
                sampling_strategy=opt.hard_neg_strategy,
                update_frequency=1,
                device='cuda'
            )
            print(f"     Memory size: {opt.hard_neg_memory_size}")
            print(f"     Top-k: {opt.hard_neg_topk}")
            print(f"     Strategy: {opt.hard_neg_strategy}")

    return L1_loss, P_loss, E_loss, D_loss, FR_loss, hard_neg_sampler  # NEW: Return hard_neg_sampler

# ============================================================================
# MODIFICATION 3: Update load_datasets to use IdentityBalancedSampler (around line 211)
# ============================================================================
def load_datasets():
    print('===> Loading datasets')

    if opt.lapaface:
        train_set = get_lapaface_training_set(opt.data_train_lapaface, size=opt.cropSize)

        # NEW: Use identity-balanced sampler if enabled
        if opt.use_identity_balanced:
            print("===> Using Identity-Balanced Batch Sampler")
            batch_sampler = IdentityBalancedSampler(
                dataset=train_set,
                batch_size=opt.batchSize,
                images_per_identity=opt.images_per_identity,
                shuffle=opt.shuffle,
                drop_last=True
            )
            training_data_loader = DataLoader(
                dataset=train_set,
                batch_sampler=batch_sampler,
                num_workers=opt.threads
            )
        else:
            # Standard random sampling
            training_data_loader = DataLoader(
                dataset=train_set,
                num_workers=opt.threads,
                batch_size=opt.batchSize,
                shuffle=opt.shuffle
            )

        test_set = get_lapaface_eval_set(opt.data_val_lapaface)
        testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)

    # Similar modifications for other datasets...

    return training_data_loader, testing_data_loader

# ============================================================================
# MODIFICATION 4: Update train function to use hard negative sampler (around line 104-129)
# ============================================================================
def train(epoch):
    model.train()
    loss_print = 0
    pic_cnt = 0
    loss_last_10 = 0
    pic_last_10 = 0
    train_len = len(training_data_loader)
    iter = 0
    torch.autograd.set_detect_anomaly(opt.grad_detect)

    for batch in tqdm(training_data_loader):
        im1, im2, path1, path2 = batch[0], batch[1], batch[2], batch[3]
        batch_size = im1.shape[0]

        # ... (validation code remains same)

        im1 = im1.cuda(non_blocking=True)
        im2 = im2.cuda(non_blocking=True)

        # Enhancement
        if opt.gamma:
            gamma = random.randint(opt.start_gamma,opt.end_gamma) / 100.0
            output_rgb = model(im1 ** gamma)
        else:
            output_rgb = model(im1)

        gt_rgb = im2
        output_hvi = model.HVIT(output_rgb)
        gt_hvi = model.HVIT(gt_rgb)
        loss_hvi = L1_loss(output_hvi, gt_hvi) + D_loss(output_hvi, gt_hvi) + E_loss(output_hvi, gt_hvi) + opt.P_weight * P_loss(output_hvi, gt_hvi)[0]
        loss_rgb = L1_loss(output_rgb, gt_rgb) + D_loss(output_rgb, gt_rgb) + E_loss(output_rgb, gt_rgb) + opt.P_weight * P_loss(output_rgb, gt_rgb)[0]

        # Add Face Recognition Perceptual Loss if enabled
        if opt.use_face_loss and FR_loss is not None:
            if batch_size > 1:
                # NEW: Use hard negative mining if enabled
                if opt.use_hard_negatives and hard_neg_sampler is not None:
                    # Extract identities from filenames
                    batch_identities = [hard_neg_sampler.extract_identity(f) for f in path2]

                    # Sample hard impostors
                    impostor_gt = hard_neg_sampler.sample_hard_impostors(
                        batch_gt=gt_rgb,
                        batch_identities=batch_identities,
                        batch_features=None  # Will be extracted inside sampler
                    )
                else:
                    # OLD: Circular shift strategy for impostor pairs
                    impostor_gt = torch.roll(gt_rgb, shifts=1, dims=0)

                # Compute discriminative multi-level face loss
                face_loss_dict = FR_loss(output_rgb, gt_rgb, impostor_gt)
                fr_loss_value = face_loss_dict['total']

                # Log face loss components every 100 iterations
                if iter % 100 == 0:
                    print(f"\n  Face Loss Components (iter {iter}):")
                    print(f"    Reconstruction: {face_loss_dict['reconstruction']:.4f}")
                    print(f"    Contrastive:    {face_loss_dict['contrastive']:.4f}")
                    print(f"    Triplet:        {face_loss_dict['triplet']:.4f}")
                    print(f"    Total:          {fr_loss_value:.4f}")
                    if opt.use_hard_negatives:
                        stats = hard_neg_sampler.get_statistics()
                        print(f"    Hard Neg Memory: {stats['memory_size']} identities")
            else:
                # Fallback for batch_size=1
                fr_loss_value = FR_loss(output_rgb, gt_rgb, gt_rgb)['total']
                if iter % 100 == 0:
                    print(f"\n  Warning: Batch size = 1, using paired loss only")

            loss = loss_rgb + opt.HVI_weight * loss_hvi + opt.FR_weight * fr_loss_value
        else:
            loss = loss_rgb + opt.HVI_weight * loss_hvi

        iter += 1

        # ... (rest of training loop remains same)

# ============================================================================
# MODIFICATION 5: Update main block to use new sampler (around line 356)
# ============================================================================
if __name__ == '__main__':
    '''
    preparision
    '''
    train_init()
    training_data_loader, testing_data_loader = load_datasets()
    model = build_model()
    optimizer,scheduler = make_scheduler()
    L1_loss, P_loss, E_loss, D_loss, FR_loss, hard_neg_sampler = init_loss()  # NEW: Unpack hard_neg_sampler

    '''
    train
    '''
    psnr = []
    ssim = []
    lpips = []
    start_epoch=0
    if opt.start_epoch > 0:
        start_epoch = opt.start_epoch
    if not os.path.exists(opt.val_folder):
        os.mkdir(opt.val_folder)

    for epoch in range(start_epoch+1, opt.nEpochs + start_epoch + 1):
        try:
            epoch_loss, pic_num = train(epoch)
        except RuntimeError as e:
            # ... error handling ...

        scheduler.step()

        # ... rest of main loop ...

# ============================================================================
# END OF MODIFICATIONS
# ============================================================================
