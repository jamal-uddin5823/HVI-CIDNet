import argparse

def option():
    # Training settings
    parser = argparse.ArgumentParser(description='CIDNet')
    parser.add_argument('--batchSize', type=int, default=8, help='training batch size')
    parser.add_argument('--cropSize', type=int, default=256, help='image crop size (patch size)')
    parser.add_argument('--nEpochs', type=int, default=1000, help='number of epochs to train for end')
    parser.add_argument('--start_epoch', type=int, default=0, help='number of epochs to start, >0 is retrained a pre-trained pth')
    parser.add_argument('--pretrained_model', type=str, default=None, help='Path to pretrained CIDNet model weights (e.g., ./weights/LOLv2_real/best_PSNR.pth)')
    parser.add_argument('--snapshots', type=int, default=10, help='Snapshots for save checkpoints pth')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate')
    parser.add_argument('--gpu_mode', type=bool, default=True)
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--threads', type=int, default=16, help='number of threads for dataloader to use')

    # choose a scheduler
    parser.add_argument('--cos_restart_cyclic', type=bool, default=False)
    parser.add_argument('--cos_restart', type=bool, default=True)

    # warmup training
    parser.add_argument('--warmup_epochs', type=int, default=3, help='warmup_epochs')
    parser.add_argument('--start_warmup', type=bool, default=True, help='turn False to train without warmup') 

    # train datasets
    parser.add_argument('--data_train_lol_blur'     , type=str, default='./datasets/LOL_blur/train')
    parser.add_argument('--data_train_lol_v1'       , type=str, default='./datasets/LOLdataset/our485')
    parser.add_argument('--data_train_lolv2_real'   , type=str, default='./datasets/LOLv2/Real_captured/Train')
    parser.add_argument('--data_train_lolv2_syn'    , type=str, default='./datasets/LOLv2/Synthetic/Train')
    parser.add_argument('--data_train_SID'          , type=str, default='./datasets/Sony_total_dark/train')
    parser.add_argument('--data_train_SICE'         , type=str, default='./datasets/SICE/Dataset/train')
    parser.add_argument('--data_train_fivek'        , type=str, default='./datasets/FiveK/train')
    parser.add_argument('--data_train_lfw'          , type=str, default='./datasets/LFW_lowlight/train')
    parser.add_argument('--data_train_lapaface'     , type=str, default='./datasets/LaPa-Face/train')

    # validation input
    parser.add_argument('--data_val_lol_blur'       , type=str, default='./datasets/LOL_blur/eval/low_blur')
    parser.add_argument('--data_val_lol_v1'         , type=str, default='./datasets/LOLdataset/eval15/low')
    parser.add_argument('--data_val_lolv2_real'     , type=str, default='./datasets/LOLv2/Real_captured/Test/Low')
    parser.add_argument('--data_val_lolv2_syn'      , type=str, default='./datasets/LOLv2/Synthetic/Test/Low')
    parser.add_argument('--data_val_SID'            , type=str, default='./datasets/Sony_total_dark/eval/short')
    parser.add_argument('--data_val_SICE_mix'       , type=str, default='./datasets/SICE/Dataset/eval/test')
    parser.add_argument('--data_val_SICE_grad'      , type=str, default='./datasets/SICE/Dataset/eval/test')
    parser.add_argument('--data_test_fivek'         , type=str, default='./datasets/FiveK/test/input')
    parser.add_argument('--data_val_lfw'            , type=str, default='./datasets/LFW_lowlight/val/low')
    parser.add_argument('--data_val_lapaface'       , type=str, default='./datasets/LaPa-Face/test')

    # validation groundtruth
    parser.add_argument('--data_valgt_lol_blur'     , type=str, default='./datasets/LOL_blur/eval/high_sharp_scaled/')
    parser.add_argument('--data_valgt_lol_v1'       , type=str, default='./datasets/LOLdataset/eval15/high/')
    parser.add_argument('--data_valgt_lolv2_real'   , type=str, default='./datasets/LOLv2/Real_captured/Test/Normal/')
    parser.add_argument('--data_valgt_lolv2_syn'    , type=str, default='./datasets/LOLv2/Synthetic/Test/Normal/')
    parser.add_argument('--data_valgt_SID'          , type=str, default='./datasets/Sony_total_dark/eval/long/')
    parser.add_argument('--data_valgt_SICE_mix'     , type=str, default='./datasets/SICE/Dataset/eval/target/')
    parser.add_argument('--data_valgt_SICE_grad'    , type=str, default='./datasets/SICE/Dataset/eval/target/')
    parser.add_argument('--data_valgt_fivek'        , type=str, default='./datasets/FiveK/test/target/')
    parser.add_argument('--data_valgt_lfw'          , type=str, default='./datasets/LFW_lowlight/val/high')
    parser.add_argument('--data_valgt_lapaface'     , type=str, default='./datasets/LaPa-Face/test/normal/')

    parser.add_argument('--val_folder', default='./results/', help='Location to save validation datasets')

    # loss weights
    parser.add_argument('--HVI_weight', type=float, default=1.0)
    parser.add_argument('--L1_weight', type=float, default=1.0)
    parser.add_argument('--D_weight',  type=float, default=0.5)
    parser.add_argument('--E_weight',  type=float, default=50.0)
    parser.add_argument('--P_weight',  type=float, default=1e-2)

    # Face Recognition Perceptual Loss (for thesis research)
    parser.add_argument('--use_face_loss', action='store_true', help='Enable face recognition perceptual loss')
    parser.add_argument('--FR_weight', type=float, default=0.5, help='Face recognition loss weight')
    parser.add_argument('--FR_model_arch', type=str, default='ir_50', choices=['ir_50', 'ir_101'],
                        help='AdaFace architecture: ir_50 or ir_101')
    parser.add_argument('--FR_model_path', type=str, default=None,
                        help='Path to pre-trained AdaFace weights (None = random init)')
    parser.add_argument('--FR_feature_distance', type=str, default='mse', choices=['mse', 'l1', 'cosine'],
                        help='Distance metric for face features')
    
    # Discriminative Multi-Level Face Loss hyperparameters
    parser.add_argument('--contrastive_margin', type=float, default=0.4,
                        help='Margin for contrastive loss (impostor separation)')
    parser.add_argument('--contrastive_weight', type=float, default=1.0,
                        help='Weight for contrastive loss component')
    parser.add_argument('--triplet_margin', type=float, default=0.2,
                        help='Margin for triplet loss')
    parser.add_argument('--triplet_weight', type=float, default=0.5,
                        help='Weight for triplet loss component')
    parser.add_argument('--face_temperature', type=float, default=0.07,
                        help='Temperature for contrastive loss (lower = harder negatives)')
    
    # use random gamma function (enhancement curve) to improve generalization
    parser.add_argument('--gamma', type=bool, default=False)
    parser.add_argument('--start_gamma', type=int, default=60)
    parser.add_argument('--end_gamma', type=int, default=120)

    # auto grad, turn off to speed up training
    parser.add_argument('--grad_detect', type=bool, default=False, help='if gradient explosion occurs, turn-on it')
    parser.add_argument('--grad_clip', type=bool, default=True, help='if gradient fluctuates too much, turn-on it')
    
    
    # choose which dataset you want to train, please only set one "True"
    parser.add_argument('--lol_v1', action='store_true', help='Train on LOL v1 dataset')
    parser.add_argument('--lolv2_real', action='store_true', help='Train on LOL v2 real dataset')
    parser.add_argument('--lolv2_syn', action='store_true', help='Train on LOL v2 synthetic dataset')
    parser.add_argument('--lol_blur', action='store_true', help='Train on LOL blur dataset')
    parser.add_argument('--SID', action='store_true', help='Train on SID dataset')
    parser.add_argument('--SICE_mix', action='store_true', help='Train on SICE mix dataset')
    parser.add_argument('--SICE_grad', action='store_true', help='Train on SICE grad dataset')
    parser.add_argument('--fivek', action='store_true', help='Train on FiveK dataset')
    parser.add_argument('--lfw', action='store_true', help='Train on LFW with synthetic low-light (for face recognition loss)')
    parser.add_argument('--lapaface', action='store_true', help='Train on LaPa-Face with underexposed images (for face recognition loss)')
    return parser
