import os
import torch
import random
from torchvision import transforms
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np
from torch.utils.data import DataLoader
from net.CIDNet import CIDNet
from data.options import option
from measure import metrics
from eval import eval
from data.data import *
from loss.losses import *
from data.scheduler import *
from tqdm import tqdm
from datetime import datetime
from loss.discriminative_face_loss import DiscriminativeMultiLevelFaceLoss

opt = option().parse_args()

def seed_torch():
    seed = random.randint(1, 1000000)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
def train_init():
    seed_torch()
    cudnn.benchmark = True
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    cuda = opt.gpu_mode
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")
    
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
        
        # Validate batch before processing
        if torch.isnan(im1).any() or torch.isinf(im1).any():
            print(f"Warning: Invalid input batch detected, skipping...")
            continue
        if torch.isnan(im2).any() or torch.isinf(im2).any():
            print(f"Warning: Invalid GT batch detected, skipping...")
            print(f"  GT range: [{im2.min()}, {im2.max()}]")
            print(f"  Files: {path1[0]}, {path2[0]}")
            continue
        
        # Clamp inputs to valid range
        im1 = torch.clamp(im1, 0, 1)
        im2 = torch.clamp(im2, 0, 1)
        
        im1 = im1.cuda()
        im2 = im2.cuda()
        
        # use random gamma function (enhancement curve) to improve generalization
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
            # Sample impostor pairs for discriminative learning
            if batch_size > 1:
                # Circular shift strategy for impostor pairs
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
            else:
                # Fallback for batch_size=1 (no impostor available)
                fr_loss_value = FR_loss(output_rgb, gt_rgb, gt_rgb)['total']
                if iter % 100 == 0:
                    print(f"\n  Warning: Batch size = 1, using paired loss only")
            
            loss = loss_rgb + opt.HVI_weight * loss_hvi + opt.FR_weight * fr_loss_value
        else:
            loss = loss_rgb + opt.HVI_weight * loss_hvi

        iter += 1
        
        if opt.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01, norm_type=2)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Store loss value and update counters
        loss_value = loss.item()
        
        # Check for NaN/Inf
        if not torch.isfinite(loss).all():
            print(f"WARNING: Non-finite loss detected at iteration {iter}")
            print(f"  loss_rgb: {loss_rgb.item() if torch.isfinite(loss_rgb) else 'NaN/Inf'}")
            print(f"  loss_hvi: {loss_hvi.item() if torch.isfinite(loss_hvi) else 'NaN/Inf'}")
            if opt.use_face_loss and 'fr_loss_value' in locals():
                print(f"  fr_loss_value: {fr_loss_value.item() if torch.isfinite(fr_loss_value) else 'NaN/Inf'}")
            print(f"  output_rgb range: [{output_rgb.min().item()}, {output_rgb.max().item()}]")
            print(f"  gt_rgb range: [{gt_rgb.min().item()}, {gt_rgb.max().item()}]")
            print(f"  input range: [{im1.min().item()}, {im1.max().item()}]")
            print(f"  Batch files: {path1[0]}, {path2[0]}")
            print(f"  Skipping this batch and continuing training...")
            continue  # Skip this batch instead of crashing
        
        loss_print = loss_print + loss_value
        loss_last_10 = loss_last_10 + loss_value
        pic_cnt += 1
        pic_last_10 += 1
        
        # Save sample images at end of epoch (before cleanup)
        if iter == train_len:
            print("===> Epoch[{}]: Loss: {:.4f} || Learning rate: lr={}.".format(epoch,
                loss_last_10/pic_last_10, optimizer.param_groups[0]['lr']))
            loss_last_10 = 0
            pic_last_10 = 0
            output_img = transforms.ToPILImage()((output_rgb)[0].squeeze(0))
            gt_img = transforms.ToPILImage()((gt_rgb)[0].squeeze(0))
            if not os.path.exists(opt.val_folder+'training'):          
                os.mkdir(opt.val_folder+'training') 
            output_img.save(opt.val_folder+'training/test.png')
            gt_img.save(opt.val_folder+'training/gt.png')
    return loss_print, pic_cnt
                

def checkpoint(epoch):
    if not os.path.exists("./weights"):          
        os.mkdir("./weights") 
    if not os.path.exists("./weights/train"):          
        os.mkdir("./weights/train")  
    model_out_path = "./weights/train/epoch_{}.pth".format(epoch)
    torch.save(model.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))
    return model_out_path
    
def load_datasets():
    print('===> Loading datasets')
    if opt.lol_v1 or opt.lol_blur or opt.lolv2_real or opt.lolv2_syn or opt.SID or opt.SICE_mix or opt.SICE_grad or opt.fivek or opt.lfw:
        if opt.lol_v1:
            train_set = get_lol_training_set(opt.data_train_lol_v1,size=opt.cropSize)
            training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=opt.shuffle)
            test_set = get_eval_set(opt.data_val_lol_v1)
            testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)

        elif opt.lol_blur:
            train_set = get_training_set_blur(opt.data_train_lol_blur,size=opt.cropSize)
            training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=opt.shuffle)
            test_set = get_eval_set(opt.data_val_lol_blur)
            testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)

        elif opt.lolv2_real:
            train_set = get_lol_v2_training_set(opt.data_train_lolv2_real,size=opt.cropSize)
            training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=opt.shuffle)
            test_set = get_eval_set(opt.data_val_lolv2_real)
            testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)

        elif opt.lolv2_syn:
            train_set = get_lol_v2_syn_training_set(opt.data_train_lolv2_syn,size=opt.cropSize)
            training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=opt.shuffle)
            test_set = get_eval_set(opt.data_val_lolv2_syn)
            testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)

        elif opt.SID:
            train_set = get_SID_training_set(opt.data_train_SID,size=opt.cropSize)
            training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=opt.shuffle)
            test_set = get_eval_set(opt.data_val_SID)
            testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)

        elif opt.SICE_mix:
            train_set = get_SICE_training_set(opt.data_train_SICE,size=opt.cropSize)
            training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=opt.shuffle)
            test_set = get_SICE_eval_set(opt.data_val_SICE_mix)
            testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)

        elif opt.SICE_grad:
            train_set = get_SICE_training_set(opt.data_train_SICE,size=opt.cropSize)
            training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=opt.shuffle)
            test_set = get_SICE_eval_set(opt.data_val_SICE_grad)
            testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)

        elif opt.fivek:
            train_set = get_fivek_training_set(opt.data_train_fivek,size=opt.cropSize)
            training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=opt.shuffle)
            test_set = get_fivek_eval_set(opt.data_val_fivek)
            testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)

        elif opt.lfw:
            train_set = get_lfw_training_set(opt.data_train_lfw, size=opt.cropSize)
            training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=opt.shuffle)
            test_set = get_lfw_eval_set(opt.data_val_lfw)
            testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)
    else:
        raise Exception("should choose a dataset")
    return training_data_loader, testing_data_loader

def build_model():
    print('===> Building model ')
    model = CIDNet().cuda()

    # Load pretrained weights if specified
    if opt.pretrained_model is not None:
        print(f'===> Loading pretrained CIDNet model from {opt.pretrained_model}')
        model.load_state_dict(torch.load(opt.pretrained_model, map_location=lambda storage, loc: storage))
        print('===> Pretrained model loaded successfully')
    elif opt.start_epoch > 0:
        pth = f"./weights/train/epoch_{opt.start_epoch}.pth"
        print(f'===> Resuming from epoch {opt.start_epoch}')
        model.load_state_dict(torch.load(pth, map_location=lambda storage, loc: storage))

    return model

def make_scheduler():
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)      
    if opt.cos_restart_cyclic:
        if opt.start_warmup:
            scheduler_step = CosineAnnealingRestartCyclicLR(optimizer=optimizer, periods=[(opt.nEpochs//4)-opt.warmup_epochs, (opt.nEpochs*3)//4], restart_weights=[1,1],eta_mins=[0.0002,0.0000001])
            scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=opt.warmup_epochs, after_scheduler=scheduler_step)
        else:
            scheduler = CosineAnnealingRestartCyclicLR(optimizer=optimizer, periods=[opt.nEpochs//4, (opt.nEpochs*3)//4], restart_weights=[1,1],eta_mins=[0.0002,0.0000001])
    elif opt.cos_restart:
        if opt.start_warmup:
            scheduler_step = CosineAnnealingRestartLR(optimizer=optimizer, periods=[opt.nEpochs - opt.warmup_epochs - opt.start_epoch], restart_weights=[1],eta_min=1e-7)
            scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=opt.warmup_epochs, after_scheduler=scheduler_step)
        else:
            scheduler = CosineAnnealingRestartLR(optimizer=optimizer, periods=[opt.nEpochs - opt.start_epoch], restart_weights=[1],eta_min=1e-7)
    else:
        raise Exception("should choose a scheduler")
    return optimizer,scheduler

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
        print(f"     Architecture: {opt.FR_model_arch}")
        print(f"     Features: {['layer2', 'layer3', 'layer4', 'fc']}")
        print(f"     Contrastive: margin={opt.contrastive_margin}, weight={opt.contrastive_weight}, temp={opt.face_temperature}")
        print(f"     Triplet: margin={opt.triplet_margin}, weight={opt.triplet_weight}")

    return L1_loss, P_loss, E_loss, D_loss, FR_loss

if __name__ == '__main__':  
    
    '''
    preparision
    '''
    train_init()
    training_data_loader, testing_data_loader = load_datasets()
    model = build_model()
    optimizer,scheduler = make_scheduler()
    L1_loss, P_loss, E_loss, D_loss, FR_loss = init_loss()
    
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
        epoch_loss, pic_num = train(epoch)
        scheduler.step()
        
        if epoch % opt.snapshots == 0:
            model_out_path = checkpoint(epoch) 
            norm_size = True

            # LOL three subsets
            if opt.lol_v1:
                output_folder = 'LOLv1/'
                label_dir = opt.data_valgt_lol_v1
            if opt.lolv2_real:
                output_folder = 'LOLv2_real/'
                label_dir = opt.data_valgt_lolv2_real
            if opt.lolv2_syn:
                output_folder = 'LOLv2_syn/'
                label_dir = opt.data_valgt_lolv2_syn
            
            # LOL-blur dataset with low_blur and high_sharp_scaled
            if opt.lol_blur:
                output_folder = 'LOL_blur/'
                label_dir = opt.data_valgt_lol_blur
                
            if opt.SID:
                output_folder = 'SID/'
                label_dir = opt.data_valgt_SID
                npy = True
            if opt.SICE_mix:
                output_folder = 'SICE_mix/'
                label_dir = opt.data_valgt_SICE_mix
                norm_size = False
            if opt.SICE_grad:
                output_folder = 'SICE_grad/'
                label_dir = opt.data_valgt_SICE_grad
                norm_size = False
                
            if opt.fivek:
                output_folder = 'fivek/'
                label_dir = opt.data_valgt_fivek
                norm_size = False

            if opt.lfw:
                output_folder = 'lfw/'
                label_dir = opt.data_valgt_lfw
                norm_size = True

            im_dir = opt.val_folder + output_folder + '**/*.png'
            
            # Clear GPU memory before evaluation
            torch.cuda.empty_cache()
            
            try:
                # Set model to eval mode before evaluation
                model.eval()
                
                eval(model, testing_data_loader, model_out_path, opt.val_folder+output_folder, 
                     norm_size=norm_size, LOL=opt.lol_v1, v2=opt.lolv2_real, alpha=0.8)
                
                # Clear GPU memory after evaluation
                torch.cuda.empty_cache()
                
                # Force garbage collection
                import gc
                gc.collect()
                
                avg_psnr, avg_ssim, avg_lpips = metrics(im_dir, label_dir, use_GT_mean=False)
                print("===> Avg.PSNR: {:.4f} dB ".format(avg_psnr))
                print("===> Avg.SSIM: {:.4f} ".format(avg_ssim))
                print("===> Avg.LPIPS: {:.4f} ".format(avg_lpips))
                psnr.append(avg_psnr)
                ssim.append(avg_ssim)
                lpips.append(avg_lpips)
                print(psnr)
                print(ssim)
                print(lpips)
                
                # Set model back to train mode
                model.train()
            except Exception as e:
                import traceback
                print(f"===> Validation failed at epoch {epoch}: {str(e)}")
                print(traceback.format_exc())
                print("===> Continuing training...")
                # Append placeholder values
                psnr.append(0.0)
                ssim.append(0.0)
                lpips.append(1.0)
                
                # Ensure model is back in train mode
                model.train()
            
            # Clear GPU memory after validation
            torch.cuda.empty_cache()
    
    now = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    with open(f"./results/training/metrics{now}.md", "w") as f:
        f.write("dataset: "+ output_folder + "\n")
        f.write(f"lr: {opt.lr}\n")
        f.write(f"batch size: {opt.batchSize}\n")
        f.write(f"crop size: {opt.cropSize}\n")
        f.write(f"HVI_weight: {opt.HVI_weight}\n")
        f.write(f"L1_weight: {opt.L1_weight}\n")
        f.write(f"D_weight: {opt.D_weight}\n")
        f.write(f"E_weight: {opt.E_weight}\n")
        f.write(f"P_weight: {opt.P_weight}\n")
        if opt.use_face_loss:
            f.write(f"FR_weight: {opt.FR_weight}\n")
            f.write(f"FR_model: {opt.FR_model_arch}\n")
            f.write(f"FR_distance: {opt.FR_feature_distance}\n")
        f.write("| Epochs | PSNR | SSIM | LPIPS |\n")
        f.write("|----------------------|----------------------|----------------------|----------------------|\n")
        for i in range(len(psnr)):
            f.write(f"| {opt.start_epoch+(i+1)*opt.snapshots} | { psnr[i]:.4f} | {ssim[i]:.4f} | {lpips[i]:.4f} |\n")  
        