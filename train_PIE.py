"""
This is the main file for training the model for PIE dataset.
"""

from models.astra_model import ASTRA_model
from models.keypoint_model import UNETEmbeddingExtractor
from utils.logger import get_logger
from utils.misc import timeit
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import torch.nn as nn
from utils.metrics import AverageMeter, BoundingBoxEvaluator
from utils.losses import Loss
from utils.misc import cxcy_to_xy, unnormalize_coords
import os
import wandb
from utils.misc import model_summary
from icecream import ic

logger = get_logger(__name__)

@timeit
def train_PIE(cfg, train_dataset, test_dataset):
    wandb.save(f'./configs/pie.yaml')
    wandb.save(f'./models/astra_model.py')
    wandb.config.update(cfg)
    # Define the training device and the number of GPUs
    device, device_list = cfg.device, cfg.device_list
    num_device = len(device_list)
    
    # Training hyperparameters
    batch_size = cfg.TRAIN.BATCH_SIZE
    num_epoch = cfg.TRAIN.NUM_EPOCH
    workers_single = cfg.TRAIN.NUM_WORKERS

    # Data loader
    train_loader = DataLoader(
        train_dataset,
        batch_size,
        num_workers=workers_single * num_device,
        shuffle=cfg.MODEL.SHUFFLE,
        pin_memory=True,
        drop_last=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size,
        num_workers=workers_single * num_device,
        shuffle=False,
        pin_memory=True,
        drop_last=True,
    )
    
    # ============ Building ASTRA Model ... ============ 
    # Define model & UNET Embedding Extractor
    model = ASTRA_model(cfg)
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
    model_n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    if cfg.MODEL.USE_PRETRAINED_UNET:
        cfg.UNET_MODE = 'testing'
        embedding_extractor.load_state_dict(torch.load('./pretrained_unet_weights/pie_unet_model_best.pt'))
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

    # ============ Model Summary ... ============
    model_summary(model, cfg)
    
    # ============ Preparing Loss Function ... ============
    # define loss function
    loss_fun = Loss(cfg)
    loss_fun = loss_fun.to(gpu_num)
    
    # ============ Preparing Optimizer ... ============
    # define optimizer
    
    param_list = list(model.parameters())
        
    if cfg.TRAIN.OPTIMIZER == "Adam":
        optimizer = torch.optim.Adam(param_list, lr=float(cfg.TRAIN.LR), weight_decay=float(cfg.TRAIN.W_DECAY))
    elif cfg.TRAIN.OPTIMIZER == "AdamW":
        optimizer = torch.optim.AdamW(param_list, lr=float(cfg.TRAIN.LR), weight_decay=float(cfg.TRAIN.W_DECAY))
    elif cfg.TRAIN.OPTIMIZER == "SGD":
        optimizer = torch.optim.SGD(param_list, lr=float(cfg.TRAIN.LR), momentum=float(cfg.TRAIN.MOMENTUM), weight_decay=float(cfg.TRAIN.W_DECAY))
        
    # ============ Learning Rate schedulers ... ============
    # define LR scheduler
    if cfg.TRAIN.LR_SCHEDULER == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=int(cfg.TRAIN.PATIENCE), min_lr=float(cfg.TRAIN.MIN_LR), 
                                                               factor = float(cfg.TRAIN.FACTOR), verbose=True)
    elif cfg.TRAIN.LR_SCHEDULER == "CosineAnnealing":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0 = 10, eta_min=float(cfg.TRAIN.MIN_LR))

    logger.info("Loss, Optimizer, AND LR scheduler are built.")

    # ============ Saving the model ... ============
    # Check if the checkpoints folder exists
    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")

    # Initialize variables for best and last models
    best_CADE = float('inf')
    best_CFDE = float('inf')
    best_ARB = float('inf')
    best_FRB = float('inf')
    best_epoch = 0
    last_model_path = os.path.join("checkpoints", f"pie_last_model.pth")
    itr_num = 0
    logger.info("Start training")
    
    # ============ Training ... ============
    for epoch in range(num_epoch):

        model.train()
        itr_num = train_one_epoch(cfg, epoch, train_loader, model, embedding_extractor, optimizer, loss_fun, itr_num, scheduler)

        # validate every certain number of epochs
        if (epoch+1) % cfg.VAL.FREQ == 0 or epoch+1 == num_epoch:
            model.eval()
            _, CADE, CFDE, ARB, FRB, MSE = val_one_epoch(cfg, epoch, test_loader, model, embedding_extractor, loss_fun, itr_num)
            
            # Scheduler
            if cfg.TRAIN.LR_SCHEDULER == "ReduceLROnPlateau":
                scheduler.step(MSE)

            # Save last and Best Model
            if cfg.MODEL.SAVE_MODEL:
                torch.save(model.state_dict(), last_model_path)

                if (CADE < best_CADE and CFDE < best_CFDE and ARB < best_ARB and FRB < best_FRB) or (best_CADE <= cfg.BEST_CADE and best_CFDE <= cfg.BEST_CFDE and best_ARB <= cfg.BEST_ARB and best_FRB <= cfg.BEST_FRB):
                    best_CADE = CADE
                    best_CFDE = CFDE
                    best_ARB = ARB
                    best_FRB = FRB
                    best_epoch = epoch+1
                    best_model_path = os.path.join("checkpoints", f"pie_CADE_{best_CADE:.3f}_CFDE_{best_CFDE:.3f}_ARB_{best_ARB:.3f}_FRB_{best_FRB:.3f}_best_model.pth")
                    torch.save(model.state_dict(), best_model_path)
                      
                    wandb.log({'Best CADE': best_CADE, 'Best CFDE': best_CFDE, 'Best ARB': best_ARB, 'Best FRB': best_FRB, 'Best Epoch': best_epoch}, step=epoch)
                    logger.info("Best model saved at epoch {}.".format(epoch+1))
                    
        wandb.log({'Learning Rate': scheduler.optimizer.param_groups[0]['lr']})
        
    logger.info("End training")
    if best_model_path:
        wandb.save(best_model_path)
        logger.info(f"Best model is from epoch {best_epoch} with CADE {best_CADE:.3f}, CFDE {best_CFDE:.3f}, ARB {best_ARB:.3f}, FRB {best_FRB:.3f}")
    wandb.finish()

def train_one_epoch(cfg, epoch, train_loader, model, embedding_extractor, optimizer, loss_fun, itr_num, scheduler=None):

    device = cfg.device
    num_device = len(cfg.device_list)
    gpu_num = device
    
    # Parallelize Model
    if num_device > 1:
        gpu_num = f'cuda:{model.device_ids[0]}'
  
    epoch_loss = AverageMeter()

    loop = tqdm(enumerate(train_loader), total= len(train_loader)) 
    for batch_idx, batch in loop:
        past_loc = batch['input_x'].unsqueeze(1).to(gpu_num)
        fut_loc = batch['target_y'].unsqueeze(1).to(gpu_num)
        itr_num += 1

        # ic(past_loc)
        ic(past_loc.shape)              # past_loc: (Batch, Agents, Frames, Obs_len)
        
        if cfg.MODEL.USE_PRETRAINED_UNET:
            # traj_coords = traj_coords.view(-1, 2)
            imgs = batch['past_images'].to(gpu_num)
            imgs = imgs.view(-1, 3, 224, 224).to(gpu_num)
            
            # Forward Prop (Embedding Extractor)
            _, _, extracted_features = embedding_extractor(imgs)
            extracted_features = extracted_features.view(*past_loc.shape[:-1], -1)
            unet_features = extracted_features.to(gpu_num)
            # ic(unet_features)
            ic(unet_features.shape)                     # unet_features: (Batch, Agents, Frames, feature_dim)
        else:
            unet_features = None
               
        past_loc = cxcy_to_xy(past_loc, cfg, gpu_num, unnormalize=False)           
        # prediction using model
        pred_traj = model(past_loc, unet_features)
       
        
        # ic(pred_traj)
        # ic(fut_loc)
        ic(pred_traj.shape)                 # pred_traj: (Batch, Agents, Frames, Obs_len)
        ic(fut_loc.shape)                   # fut_loc: (Batch, Agents, Frames, Pred_len)
        
        # loss calculation
        pred_traj = unnormalize_coords(pred_traj, cfg, gpu_num)
        fut_loc = cxcy_to_xy(fut_loc, cfg, gpu_num)
        loss = loss_fun(pred_traj, fut_loc, cfg)
            
        # zero_grad, backpropagation, and step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if cfg.TRAIN.LR_SCHEDULER == "CosineAnnealing":
            scheduler.step(epoch + batch_idx / len(train_loader))
        
        # log
        torch.cuda.synchronize()
        epoch_loss.update(loss.item())
        
        loop.set_description(f"Epoch [{epoch+1}/{cfg.TRAIN.NUM_EPOCH}]")
        loop.set_postfix(loss=epoch_loss.avg)

    wandb.log({'Train Loss': epoch_loss.avg}, step=epoch) # for wandb
    
    return itr_num

def val_one_epoch(cfg, epoch, val_loader, model, embedding_extractor, loss_fun, itr_num):
    
    device = cfg.device
    num_device = len(cfg.device_list)
    gpu_num = device
    
    # Parallelize Model
    if num_device > 1:
        gpu_num = f'cuda:{model.device_ids[0]}'
 
    epoch_loss = AverageMeter()
    centre_fde_metric = AverageMeter()
    centre_ade_metric = AverageMeter()
    arb_metric = AverageMeter()
    frb_metric = AverageMeter()
    mse_metric = AverageMeter()

    with torch.no_grad():
        loop = tqdm(enumerate(val_loader), total= len(val_loader)) 
        for batch_idx, batch in loop:
            past_loc = batch['input_x'].unsqueeze(1).to(gpu_num)
            fut_loc = batch['target_y'].unsqueeze(1).to(gpu_num)
            itr_num += 1
            
            # UNET Feature Extractor
            if cfg.MODEL.USE_PRETRAINED_UNET:
                imgs = batch['past_images'].to(gpu_num)
                # traj_coords = traj_coords.view(-1, 2)
                imgs = imgs.view(-1, 3, 224, 224).to(gpu_num)
                
                # Forward Prop (Embedding Extractor)
                _, _, extracted_features = embedding_extractor(imgs)
                extracted_features = extracted_features.view(*past_loc.shape[:-1], -1)  
                unet_features = extracted_features.to(gpu_num)
                # unet_features = unet_features.detach()
            else:
                unet_features = None
            past_loc = cxcy_to_xy(past_loc, cfg, gpu_num, unnormalize=False)
            # prediction using model
            pred_traj = model(past_loc, unet_features)
                
            # loss calculation
            pred_traj = unnormalize_coords(pred_traj, cfg, gpu_num)
            fut_loc = cxcy_to_xy(fut_loc, cfg, gpu_num)
            loss = loss_fun(pred_traj, fut_loc, cfg)
            
            # Evaluation metrics
            evaluator = BoundingBoxEvaluator(pred_traj, fut_loc)
            centre_fde_values = evaluator.calculate_center_fde()
            centre_ade_values = evaluator.calculate_center_ade()
            arb_values = evaluator.calculate_arb()
            frb_values = evaluator.calculate_frb()
            mse_values = evaluator.calculate_mse()          

            # log
            epoch_loss.update(loss.item())
            centre_fde_metric.update(centre_fde_values.mean().item())
            centre_ade_metric.update(centre_ade_values.mean().item())
            arb_metric.update(arb_values.mean().item())
            frb_metric.update(frb_values.mean().item())
            mse_metric.update(mse_values.mean().item())
            
            loop.set_description(f"Val epoch [{epoch+1}]")
            loop.set_postfix(CADE=centre_ade_metric.avg, CFDE=centre_fde_metric.avg, 
                             MSE=mse_metric.avg, ARB=arb_metric.avg, FRB=frb_metric.avg, loss=epoch_loss.avg, )

    # val_metrics['Ego_loss'] = epoch_ego_loss.avg
    wandb.log({'Val Loss': epoch_loss.avg}, step=epoch)
    wandb.log({'CADE': centre_ade_metric.avg}, step=epoch)
    wandb.log({'CFDE': centre_fde_metric.avg}, step=epoch)
    wandb.log({'MSE': mse_metric.avg}, step=epoch)
    wandb.log({'ARB': arb_metric.avg}, step=epoch)
    wandb.log({'FRB': frb_metric.avg}, step=epoch)
    wandb.log({'Epoch': epoch}, step=epoch)

    return epoch_loss.avg, centre_ade_metric.avg, centre_fde_metric.avg, arb_metric.avg, frb_metric.avg, mse_metric.avg