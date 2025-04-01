"""
This is the main file for training the model for ETH-UCY dataset.
"""

from models.astra_model import ASTRA_model
from models.keypoint_model import UNETEmbeddingExtractor
from utils.logger import get_logger
from utils.misc import timeit
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import torch.nn as nn
from utils.metrics import AverageMeter, TrajectoryEvaluator
import os
import wandb
from utils.losses import Loss, KLDivergenceLoss, DiversityLoss, GaussianKLDivergenceLoss
from icecream import ic
from utils.misc import model_summary

logger = get_logger(__name__)

@timeit
def train_ETH(cfg, train_dataset, test_dataset):
    wandb.save(f'./configs/{cfg.SUBSET}.yaml')
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
    
    # ============ Model Summary ... ============
    model_summary(model, cfg)
   
    # ============ Preparing Loss Function ... ============
    # define loss function
    kl_divergence = KLDivergenceLoss()
    kl_divergence = kl_divergence.to(gpu_num)
    diversity_fun = DiversityLoss()
    diversity_fun = diversity_fun.to(gpu_num)
    gaussian_kl = GaussianKLDivergenceLoss()
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
    best_loss = float('inf')
    best_fde = float('inf')
    best_ade = float('inf')
    best_epoch = 0
    last_model_path = os.path.join("checkpoints", f"{cfg.SUBSET}_last_model.pth")
    itr_num = 0
    logger.info("Start training")
    
    # ============ Training ... ============
    for epoch in range(num_epoch):

        model.train()
        itr_num = train_one_epoch(cfg, epoch, train_loader, model, 
                                  embedding_extractor, optimizer, loss_fun, 
                                  kl_divergence, diversity_fun, gaussian_kl, itr_num, 
                                  scheduler)

        # validate every certain number of epochs
        if (epoch+1) % cfg.VAL.FREQ == 0 or epoch+1 == num_epoch:
            model.eval()
            val_loss, ade, fde = val_one_epoch(cfg, epoch, test_loader, model, embedding_extractor, loss_fun, itr_num)
            
            # Scheduler
            if cfg.TRAIN.LR_SCHEDULER == "ReduceLROnPlateau":
                scheduler.step(val_loss)

            # Save last and Best Model
            if cfg.MODEL.SAVE_MODEL:
                torch.save(model.state_dict(), last_model_path)

                if (ade <= cfg.BEST_ADE and fde <= cfg.BEST_FDE):
                    
                    # best_model_path = os.path.join("checkpoints", f"{cfg.SUBSET}_ADE_{best_ade:.3f}_FDE_{best_fde:.3f}_best_model.pth")
                    # torch.save(model.state_dict(), best_model_path)
                       
                    # wandb.log({'Best ADE': best_ade, 'Best FDE': best_fde, 'Best Epoch': best_epoch}, step=epoch)
                    logger.info("Best model saved at epoch {}.".format(epoch+1))

                if (ade < best_ade and fde < best_fde):
                    best_loss = val_loss
                    best_ade = ade
                    best_fde = fde
                    best_epoch = epoch+1
                    best_model_path = os.path.join("checkpoints", f"{cfg.SUBSET}_ADE_{best_ade:.3f}_FDE_{best_fde:.3f}_best_model.pth")
                    torch.save(model.state_dict(), best_model_path)
                       
                    wandb.log({'Best ADE': best_ade, 'Best FDE': best_fde, 'Best Epoch': best_epoch}, step=epoch)
                    logger.info("Best model saved at epoch {}.".format(epoch+1))

                
                    
        wandb.log({'Learning Rate': scheduler.optimizer.param_groups[0]['lr']})
        
    logger.info("End training")
    if best_model_path:
        wandb.save(best_model_path)
        logger.info("Best model is from epoch {} with loss {:.3f}, ADE {:.3f}, FDE {:.3f}.".format(best_epoch, best_loss, best_ade, best_fde))
    wandb.finish()

def train_one_epoch(cfg, epoch, train_loader, model, embedding_extractor, 
                    optimizer, loss_fun, kl_divergence, diversity_fun, gaussian_kl,
                    itr_num, scheduler=None):

    device = cfg.device
    num_device = len(cfg.device_list)
    gpu_num = device
    
    # Parallelize Model
    if num_device > 1:
        gpu_num = f'cuda:{model.device_ids[0]}'
  
    traj_loss = AverageMeter()
    kl_loss = AverageMeter()
    epoch_loss = AverageMeter()
    diversity_loss = AverageMeter()
    traj_c_loss = AverageMeter()

    loop = tqdm(enumerate(train_loader), total= len(train_loader)) 
    for batch_idx, batch in loop:
        past_loc, fut_loc, num_valid, imgs, gt_maps, traj_coords = batch
        itr_num += 1

        if cfg.MODEL.USE_PRETRAINED_UNET:
            traj_coords = traj_coords.view(-1, 2)
            imgs = imgs.view(-1, 3, 224, 224).to(gpu_num)
            
            # Forward Prop (Embedding Extractor)
            _, _, extracted_features = embedding_extractor(imgs)
            extracted_features = extracted_features.view(*past_loc.shape[:-1], -1)
            unet_features = extracted_features.to(gpu_num)
            
            # ic(unet_features)     
            ic(unet_features.shape)              # unet_features: (Batch, Agents, Frames, feature_dim)
            
        else:
            unet_features = None
            
        # data, target to device 
        past_loc = past_loc.to(gpu_num)      
        fut_loc = fut_loc.to(gpu_num)       
        num_valid = num_valid.to(gpu_num)    
        num_valid = num_valid.type(torch.int)

        # Normalizing
        mean_past_traj = past_loc.mean(dim = -2, keepdims = True)
        past_loc = past_loc - mean_past_traj
        fut_loc = fut_loc - mean_past_traj

        # ic(past_loc)
        ic(past_loc.shape)                  # past_loc: (Batch, Agents, Frames, 2)
    
        # prediction using model
        mean, log_var, K_pred_traj, mean_c, log_var_c, C_pred_traj = model(past_loc, fut_loc, unet_features)
        # mean, log_var, K_pred_traj = model(past_loc, fut_loc, unet_features)
        # ic(pred_traj)
        # ic(fut_loc)
        ic(K_pred_traj.shape)                 # pred_traj: (Batch, Agents, Frames, 2)
        # pred_traj = K_pred_traj.mean(dim = -3)
        # ic(pred_traj.shape) 
        ic(fut_loc.shape)                   # fut_loc: (Batch, Agents, Frames, 2)

        # loss calculation
        loss = loss_fun(K_pred_traj, fut_loc, cfg, True)
        loss_c = loss_fun(C_pred_traj, fut_loc, cfg) if cfg.MODEL.USE_VAE else torch.tensor(0.)
        
        if cfg.MODEL.USE_VAE:
            # kl_div_loss = kl_divergence(mean_c, log_var_c)
            # div_loss = diversity_fun(K_pred_traj, gpu_num) 
            kl_div_loss = gaussian_kl(mean, log_var, mean_c, log_var_c)
        else:
            kl_div_loss = torch.tensor(0.)
            div_loss = torch.tensor(0.)

        sampling_loss = loss # + 0.01*div_loss
        elbo_loss = loss_c + 0.1*kl_div_loss
        final_loss = sampling_loss + elbo_loss

        # zero_grad, backpropagation, and step
        optimizer.zero_grad()
        final_loss.backward()
        optimizer.step()
        
        if cfg.TRAIN.LR_SCHEDULER == "CosineAnnealing":
            scheduler.step(epoch + batch_idx / len(train_loader))
        
        # log
        torch.cuda.synchronize()
        if cfg.MODEL.USE_VAE:
            kl_loss.update(kl_div_loss.item())
        traj_c_loss.update(loss_c.item())
        traj_loss.update(loss.item())
        epoch_loss.update(final_loss.item())
        # diversity_loss.update(div_loss.item())

        loop.set_description(f"Epoch [{epoch+1}/{cfg.TRAIN.NUM_EPOCH}]")
        loop.set_postfix(loss=epoch_loss.avg, loss_c = traj_c_loss.avg , traj_loss=traj_loss.avg, kl_loss=kl_loss.avg)
        
    wandb.log({'Train/loss': epoch_loss.avg}, step=epoch) 
    if cfg.MODEL.USE_VAE:
        wandb.log({'Metrics/KL_loss': kl_loss.avg}, step=epoch)
        wandb.log({'Metrics/traj_loss': traj_loss.avg}, step=epoch)
    
    return itr_num

def val_one_epoch(cfg, epoch, val_loader, model, embedding_extractor, loss_fun, itr_num):
    
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
        loop = tqdm(enumerate(val_loader), total= len(val_loader)) 
        for batch_idx, batch in loop:
            past_loc, fut_loc, num_valid, imgs, gt_maps, traj_coords = batch
            itr_num += 1
            
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

            # Normalizing
            mean_past_traj = past_loc.mean(dim = -2, keepdims = True)
            past_loc = past_loc - mean_past_traj
            fut_loc = fut_loc - mean_past_traj

            # prediction using model
            _, _, K_pred_traj = model(past_loc, None, unet_features, mode = 'testing')

            # loss calculation
            loss = loss_fun(K_pred_traj, fut_loc, cfg, True)

            # Denormalizing
            k_mean_past_traj = torch.unsqueeze(mean_past_traj, dim = -3)
            K_pred_traj = K_pred_traj + k_mean_past_traj
            fut_loc = fut_loc + mean_past_traj
            # Evaluation metrics
            evaluator = TrajectoryEvaluator(cfg, K_pred_traj, fut_loc)
            ade_values = evaluator.calculate_ade()  # (batch_size, num_agents) 
            fde_values = evaluator.calculate_fde()  # (batch_size, num_agents)           

            # log
            epoch_loss.update(loss.item())
            ade_metric.update(ade_values.mean().item())
            fde_metric.update(fde_values.mean().item()) # fde_values.item()
            
            loop.set_description(f"Val epoch [{epoch+1}]")
            if cfg.MODEL.USE_VAE:
                loop.set_postfix(Loss=epoch_loss.avg, minADE=ade_metric.avg, minFDE=fde_metric.avg)
            else:
                loop.set_postfix(Loss=epoch_loss.avg, ADE=ade_metric.avg, FDE=fde_metric.avg) 
 

    # val_metrics['Ego_loss'] = epoch_ego_loss.avg
    wandb.log({'Val/Ego_loss': epoch_loss.avg}, step=epoch)
    if cfg.MODEL.USE_VAE:
        wandb.log({'Metrics/minADE': ade_metric.avg}, step=epoch)
        wandb.log({'Metrics/minFDE': fde_metric.avg}, step=epoch)
    else:
        wandb.log({'Metrics/ADE': ade_metric.avg}, step=epoch)
        wandb.log({'Metrics/FDE': fde_metric.avg}, step=epoch)
    wandb.log({'Epoch': epoch}, step=epoch)

    return epoch_loss.avg, ade_metric.avg, fde_metric.avg