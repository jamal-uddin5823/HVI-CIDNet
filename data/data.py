from torchvision.transforms import Compose, ToTensor, RandomCrop, RandomHorizontalFlip, RandomVerticalFlip, Resize
from data.LOLdataset import *
from data.eval_sets import *
from data.SICE_blur_SID import *
from data.fivek import *
from data.lfw_dataset import *

def transform1(size=256):
    return Compose([
        RandomCrop((size, size)),
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        ToTensor(),
    ])

def transform_lfw(size=256):
    """Transform for LFW dataset - resize first since LFW images are typically smaller"""
    return Compose([
        Resize((size + 32, size + 32)),  # Resize to slightly larger than crop size
        RandomCrop((size, size)),
        RandomHorizontalFlip(),
        ToTensor(),  # No vertical flip for faces
    ])

def transform2():
    return Compose([ToTensor()])



def get_lol_training_set(data_dir,size):
    return LOLDatasetFromFolder(data_dir, transform=transform1(size))


def get_lol_v2_training_set(data_dir,size):
    return LOLv2DatasetFromFolder(data_dir, transform=transform1(size))


def get_training_set_blur(data_dir,size):
    return LOLBlurDatasetFromFolder(data_dir, transform=transform1(size))


def get_lol_v2_syn_training_set(data_dir,size):
    return LOLv2SynDatasetFromFolder(data_dir, transform=transform1(size))


def get_SID_training_set(data_dir,size):
    return SIDDatasetFromFolder(data_dir, transform=transform1(size))


def get_SICE_training_set(data_dir,size):
    return SICEDatasetFromFolder(data_dir, transform=transform1(size))

def get_SICE_eval_set(data_dir):
    return SICEDatasetFromFolderEval(data_dir, transform=transform2())

def get_eval_set(data_dir):
    return DatasetFromFolderEval(data_dir, transform=transform2())

def get_fivek_training_set(data_dir,size):
    return FiveKDatasetFromFolder(data_dir, transform=transform1(size))

def get_fivek_eval_set(data_dir):
    return SICEDatasetFromFolderEval(data_dir, transform=transform2())

def get_lfw_training_set(data_dir, size):
    return LFWDatasetFromFolder(data_dir, transform=transform_lfw(size))

def get_lfw_eval_set(data_dir):
    # For eval, resize to standard size without random crop
    return LFWDatasetFromFolderEval(data_dir, transform=Compose([
        Resize((256, 256)),  # Resize to standard size for evaluation
        ToTensor()
    ]))