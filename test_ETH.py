
import argparse
import sys
import yaml
import munch
from utils.logger import create_exp_name, get_logger
from utils.misc import set_seed, set_device
import socket
import getpass
import torch
from torch import nn
from data.eth import ETH_dataset
import torch.utils.data as data
from utils.metrics import AverageMeter, TrajectoryEvaluator
from pprint import pprint
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from icecream import ic
from models.astra_model import ASTRA_model
from models.keypoint_model import UNETEmbeddingExtractor
from tqdm import tqdm
ic.disable()

logger = get_logger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train transformer models.")
    parser.add_argument(
        "--config_file",
        default="configs/eth.yaml",
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
    logger.info("Subset Name: %s", cfg.SUBSET)

    # Set up the training device and the number of GPUs
    device, device_list = set_device(cfg.TRAIN.DEVICE, cfg.TRAIN.BATCH_SIZE)
    cfg.update(device = device)
    cfg.update(device_list = device_list)
    num_device = len(device_list)
    
    # set manual seed for reproducibility
    set_seed(cfg.TRAIN.SEED)
    reshape_size = cfg.DATA.MIN_RESHAPE_SIZE
    mean = cfg.DATA.MEAN
    std = cfg.DATA.STD
    batch_size = cfg.TRAIN.BATCH_SIZE
    num_epoch = cfg.TRAIN.NUM_EPOCH
    workers_single = cfg.TRAIN.NUM_WORKERS

    transforms = A.Compose([A.LongestMaxSize(reshape_size),
                            A.PadIfNeeded(reshape_size,reshape_size, border_mode=cv2.BORDER_CONSTANT, value = 0),
                            A.Normalize(mean, std, max_pixel_value=255.0),
                            ToTensorV2()], keypoint_params=A.KeypointParams(format='yx'))

    # ============ preparing dataset ... ============

    test_dataset = ETH_dataset(cfg, mode='testing', img_transforms=transforms)
    logger.info('Done Loading Testing Dataset: {:,} testing samples: '.format(len(test_dataset)))
    test_loader = data.DataLoader(
                  test_dataset,
                  batch_size,
                  num_workers=workers_single * num_device,
                  shuffle=False,
                  pin_memory=True,
                  drop_last=True,
                )

    # ============ building model ... ============

    model = ASTRA_model(cfg)
    model.load_state_dict(torch.load(f'./pretrained_astra_weights/{cfg.SUBSET}_best_model.pth'))
    logger.info("ASTRA Model is built.")
    
    if cfg.MODEL.USE_PRETRAINED_UNET:
        embedding_extractor = UNETEmbeddingExtractor(cfg)
        logger.info("Using Pretrained U-Net Embedding Extractor.")
    
    gpu_num = device
    # parallelize model
    if num_device > 1:
        model = nn.DataParallel(model, device_ids = device_list)
        gpu_num = f'cuda:{model.device_ids[0]}'  
    model = model.to(device)
    
    if cfg.MODEL.USE_PRETRAINED_UNET:
        cfg.UNET_MODE = 'testing'
        embedding_extractor.load_state_dict(torch.load('./pretrained_unet_weights/eth_unet_model_best.pt'))
        embedding_extractor.unet.decoder = nn.Identity()
        embedding_extractor.feature_extractor = nn.Identity()
        embedding_extractor.seg_head = nn.Identity()
        embedding_extractor.branch1 = nn.Identity()
        embedding_extractor.branch2 = nn.Identity()
        embedding_extractor.regression_head = nn.Identity()
        for param in embedding_extractor.parameters():
            param.requires_grad = False
        embedding_extractor.eval()
        embedding_extractor = embedding_extractor.to(device)
    else:
        embedding_extractor = None

    # ============ testing model ... ============

    device = cfg.device
    num_device = len(cfg.device_list)
    gpu_num = device
    
    # Parallelize Model
    if num_device > 1:
        gpu_num = f'cuda:{model.device_ids[0]}'
 
    epoch_loss = AverageMeter()
    ade_metric = AverageMeter()
    fde_metric = AverageMeter()

    with torch.no_grad():
        loop = tqdm(enumerate(test_loader), total= len(test_loader), leave = False) 
        for batch_idx, batch in loop:
            past_loc, fut_loc, num_valid, imgs, gt_maps, traj_coords = batch         
            # UNET Feature Extractor
            if cfg.MODEL.USE_PRETRAINED_UNET:
                traj_coords = traj_coords.view(-1, 2)
                imgs = imgs.view(-1, 3, 224, 224).to(gpu_num)
                
                # Forward Prop (Embedding Extractor)
                _, _, extracted_features = embedding_extractor(imgs)
                extracted_features = extracted_features.view(*past_loc.shape[:-1], -1)  
                unet_features = extracted_features.to(gpu_num)
            else:
                unet_features = None
                
            # data, target to device
            past_loc = past_loc.to(gpu_num)  
            fut_loc = fut_loc.to(gpu_num)       
            num_valid = num_valid.to(gpu_num)    
            num_valid = num_valid.type(torch.int)
                      
            # prediction using model
            pred_traj = model(past_loc, unet_features)
            
            # Evaluation metrics
            evaluator = TrajectoryEvaluator(pred_traj, fut_loc)
            ade_values = evaluator.calculate_ade()
            fde_values = evaluator.calculate_fde()         

            # log
            ade_metric.update(ade_values.mean().item())
            fde_metric.update(fde_values.mean().item())
            
            loop.set_postfix(ADE=ade_metric.avg, FDE=fde_metric.avg)         
    metrics = {
    'ADE:': ade_metric.avg,
    'FDE:': fde_metric.avg
    }
    print('-' * 30) 
    print('Final Metrics:') 
    print('-' * 30) 
    for metric, value in metrics.items():
        print(f'| {metric.ljust(3)} | {value:.5f} |') 
    print('-' * 30)