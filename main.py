
import argparse
import sys
import yaml
import munch
from utils.logger import create_exp_name, get_logger
from utils.misc import set_seed, set_device
import socket
import getpass
from train_ETH import train_ETH
from train_PIE import train_PIE
import torch
from data.eth import ETH_dataset
import torch.utils.data as data
from data.pie_data_layer import PIEDataLayer
from pprint import pprint
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from icecream import ic
import wandb

# ic.disable()                # Enable (comment) to enable debugging

# wandb.init(name='univ_exp_pen')
wandb.init(mode='dryrun')

logger = get_logger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train transformer models.")
    parser.add_argument(
        "--config_file",
        default="configs/univ.yaml",
        type=str,
        help="path to config file.",
    )
    args = parser.parse_args()

    # Load config yaml file as nested object
    cfg = yaml.safe_load(open(args.config_file, "r"))
    cfg = munch.munchify(cfg)
    
    # start logging
    name = create_exp_name(cfg)
    logger.info("Host name: %s", socket.gethostname())
    logger.info("User name: %s", getpass.getuser())
    logger.info("Python Version: %s", sys.version)
    logger.info("PyTorch Version: %s", torch.__version__)
    logger.info("Experiment Name: %s", name)
    logger.info("Dataset Name: %s", cfg.DATASET)
    if cfg.DATASET == 'ETH_UCY':
        logger.info("Subset Name: %s", cfg.SUBSET)
    logger.info("Config: %s", pprint(cfg))

    # Set up the training device and the number of GPUs
    device, device_list = set_device(cfg.TRAIN.DEVICE, cfg.TRAIN.BATCH_SIZE)
    cfg.update(device = device)
    cfg.update(device_list = device_list)
    
    # set manual seed for reproducibility
    set_seed(cfg.TRAIN.SEED)
    # torch.set_default_dtype(torch.float16)
    reshape_size = cfg.DATA.MIN_RESHAPE_SIZE
    mean = cfg.DATA.MEAN
    std = cfg.DATA.STD
    train_transforms = A.Compose([A.LongestMaxSize(reshape_size),
                            A.PadIfNeeded(reshape_size,reshape_size, border_mode=cv2.BORDER_CONSTANT, value = 0),
                            A.Normalize(mean, std, max_pixel_value=255.0),
                            ToTensorV2()], keypoint_params=A.KeypointParams(format='yx'))

    # ============ preparing dataset ... ============
    if cfg.DATASET == 'ETH_UCY':
        train_dataset = ETH_dataset(cfg, mode='training', img_transforms=train_transforms)
        logger.info('Done Loading Training Dataset: {:,} training samples: '.format(len(train_dataset)))
        
        test_dataset = ETH_dataset(cfg, mode='testing', img_transforms=train_transforms)
        logger.info('Done Loading Testing Dataset: {:,} testing samples: '.format(len(test_dataset)))
        
        # ============ training model ... ============
        train_ETH(cfg, train_dataset, test_dataset)

    
    elif cfg.DATASET == 'PIE':
        def my_collate_fn(batch):
            return batch[0]

        train_dataset = PIEDataLayer(cfg, cfg.MODE, train_transforms)
        logger.info('Done Loading Testing Dataset: {:,} testing samples: '.format(len(train_dataset)))
        test_dataset = PIEDataLayer(cfg, 'test', train_transforms)
        logger.info('Done Loading Testing Dataset: {:,} testing samples: '.format(len(test_dataset)))
        print(cfg.TRAIN.BATCH_SIZE)
        data_loaders = data.DataLoader(
                dataset=train_dataset,
                batch_size=cfg.TRAIN.BATCH_SIZE,
                shuffle=cfg.MODE=='train',
                num_workers=cfg.TRAIN.NUM_WORKERS)
        
        # ============ training model ... ============
        train_PIE(cfg, train_dataset, test_dataset)
        
    else:
        logger.info('Dataset not found!')

    
