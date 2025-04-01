import yaml
import munch
from utils.misc import set_seed, set_device, fetch_coords_from_map
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import transforms
import numpy as np
from tqdm import tqdm
import pathlib
import os
from utils.misc import unnormalize_coords, cxcy_to_xy, fetch_coords_from_map
import pandas as pd
import matplotlib.pyplot as plt
from utils.losses import WeightedHausdorffDistance
from models.keypoint_model import UNETEmbeddingExtractor
import cv2
from pathlib import Path

device = 'cuda' if torch.cuda.is_available() else 'cpu'

args = dict()
args["config_file"] = "configs/pie.yaml"
# Load config yaml file as nested object
cfg = yaml.safe_load(open(args["config_file"], "r"))
cfg = munch.munchify(cfg)
cfg.UNET_MODE = 'training'
cfg.TRAIN.DEVICE

device, device_list = set_device(cfg.TRAIN.DEVICE, cfg.TRAIN.BATCH_SIZE)
cfg.update(device = device)
cfg.update(device_list = device_list)

set_seed(cfg.TRAIN.SEED)

pretrain_emb_path = Path('./pretrained_keypoint_embeddings/pie_keypoint_embeddings.pt')
pretrain_emb_path.parent.mkdir(parents=True, exist_ok=True)

class PIE_Images_dataset(Dataset):
    def __init__(self, cfg, base_dir, df_data, unique_data_frame_info, img_transforms = None):
        self.base_dir = base_dir
        self.labels_df = df_data
        self.unique_data_frame_info = unique_data_frame_info

        # Setting variables to get all labels for a given frame and to convert to pixel coords
        self.img_transforms = img_transforms
        self.reshape_size = cfg.DATA.MIN_RESHAPE_SIZE

    def __len__(self):
        return len(self.unique_data_frame_info)

    def __getitem__(self, idx):
        set_id = int(self.unique_data_frame_info.iloc[idx]['set_id'])
        video_id = int(self.unique_data_frame_info.iloc[idx]['video_id'])
        frame_no = int(self.unique_data_frame_info.iloc[idx]['frame_no'])
        img_path = self.base_dir/f'images/set{set_id:02d}/video_{video_id:04d}/{frame_no:05d}.png'
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Fetching Pixel coordinate for current frame
        label = self.labels_df[(self.labels_df['set_id'] == set_id) & \
                            (self.labels_df['video_id'] == video_id) & \
                            (self.labels_df['frame_no'] == frame_no)]
        xy_coord = label[['x_1', 'y_1', 'x_2', 'y_2']].values
        pixel_coord = ((xy_coord[:,:2] + xy_coord[:,2:])/2).astype(int)
        h, w = img.shape[:-1]
        # Flip
        temp = pixel_coord[:,0].copy()
        pixel_coord[:,0] = pixel_coord[:,1]
        pixel_coord[:,1] = temp
        if pixel_coord.size == 0:
            pixel_coord = np.array([[h,w]])
        if self.img_transforms:
            # Check if the pixel Coord is out of range then set it to -1, -1
            # Since the agent is not properly visible in the frame
            
            out_of_range_index = np.where(pixel_coord >= (h,w))[0]
            pixel_coord[out_of_range_index,:] = (h-1, w-1)
            transformed = self.img_transforms(image=img, keypoints=pixel_coord)
            img = transformed['image']
            pixel_coord = transformed['keypoints']
            pixel_coord = np.array(pixel_coord)
            if out_of_range_index.size != 0:
                pixel_coord[out_of_range_index,:] = (-1, -1)
            pixel_coord = pixel_coord.astype(np.int32)

        output_img_keypoint = torch.zeros(*img.shape[-2:])
        no_coords = np.where(pixel_coord != (-1,-1))[0]
        output_img_keypoint[pixel_coord[no_coords,0], pixel_coord[no_coords,1]] = 1
        return img, output_img_keypoint
                
def verify_all_images_exists(base_dir,all_imgs, df):
    df_final = df.copy()
    for idx in tqdm(range(len(df)), leave=False):
        set_id = int(df.iloc[idx]['set_id'])
        video_id = int(df.iloc[idx]['video_id'])
        frame_no = int(df.iloc[idx]['frame_no'])
        img_path = base_dir/f'images/set{set_id:02d}/video_{video_id:04d}/{frame_no:05d}.png'
        if img_path not in all_imgs:
            df_final.drop(df.index[idx], inplace = True)
    return df_final

base_dir = pathlib.Path('datasets/PIE/')
all_imgs = set(base_dir.glob('images/*/*/*.png'))

labels_df = pd.read_csv(base_dir/'final_annotations.csv')
unique_data_frame_info = labels_df[['set_id', 'video_id', 'frame_no']].drop_duplicates()
unique_data_frame_info = verify_all_images_exists(base_dir, all_imgs, unique_data_frame_info)

unique_data_frame_info = unique_data_frame_info.sample(frac=1, ignore_index=True)
train_split = 0.9
train_img_data_frame_info = unique_data_frame_info.iloc[:int(train_split*len(unique_data_frame_info))]
valid_img_data_frame_info = unique_data_frame_info.iloc[int(train_split*len(unique_data_frame_info)):]

reshape_size = cfg.DATA.MIN_RESHAPE_SIZE
mean = cfg.DATA.MEAN
std = cfg.DATA.STD

invTransform = transforms.Compose([transforms.Normalize(mean = [0., 0., 0.], std = [1/std[0], 1/std[1], 1/std[2]]),
                                  transforms.Normalize(mean = [-mean[0], -mean[1], -mean[2]], std = [1., 1., 1.])])
train_transforms = A.Compose([A.LongestMaxSize(reshape_size),
                            A.PadIfNeeded(reshape_size,reshape_size, border_mode=cv2.BORDER_CONSTANT, value = 0),
                            A.Normalize(mean, std, max_pixel_value=255.0),
                            ToTensorV2()], keypoint_params=A.KeypointParams(format='yx'))

valid_transforms = A.Compose([A.LongestMaxSize(reshape_size),
                            A.PadIfNeeded(reshape_size,reshape_size, border_mode=cv2.BORDER_CONSTANT, value = 0),
                            A.Normalize(mean, std, max_pixel_value=255.0),
                            ToTensorV2()], keypoint_params=A.KeypointParams(format='yx'))

train_dataset = PIE_Images_dataset(cfg, base_dir, labels_df, train_img_data_frame_info, img_transforms=train_transforms)
valid_dataset = PIE_Images_dataset(cfg, base_dir, labels_df, valid_img_data_frame_info, img_transforms=valid_transforms)
device, device_list = cfg.device, cfg.device_list
num_device = len(device_list)

# Training hyperparameters
batch_size = 64
num_epoch = cfg.TRAIN.NUM_EPOCH
workers_single = cfg.TRAIN.NUM_WORKERS
learning_rate = float(cfg.TRAIN.LR)

# Data loader
train_loader = DataLoader(
    train_dataset,
    batch_size,
    num_workers=workers_single * num_device,
    shuffle=False,
    pin_memory=True,
    drop_last=True,
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size,
    num_workers=workers_single * num_device,
    shuffle=False,
    pin_memory=True,
    drop_last=True,
)

model = UNETEmbeddingExtractor(cfg)
gpu_num = device
if num_device > 1:
    model = nn.DataParallel(model, device_ids = device_list)
    gpu_num = f'cuda:{model.device_ids[0]}'
model = model.to(device)
loss_weighted_hausdroff = WeightedHausdorffDistance(reshape_size, reshape_size, device=device)
loss_weighted_hausdroff = loss_weighted_hausdroff.to(gpu_num)
optim = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0005)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=0.1, 
                                                       patience=3, verbose=True, threshold=0.0001, 
                                                       threshold_mode='rel', cooldown=0, min_lr=0, 
                                                       eps=1e-08)


print("Training U-Net for Keypoint Embeddings...")
epochs = 2
best_valid_loss = 1000000.0
for epoch in range(epochs):
    loop = tqdm(enumerate(train_loader), total= len(train_loader), leave=False) 
    loop.set_description(f"Epoch [{epoch+1}/{epochs}]")
    epoch_loss = 0.0
    model.train()
    for (idx, batch) in loop:
        optim.zero_grad()
        imgs, gt_maps = batch
        gt_maps = gt_maps.view(-1, 224, 224)

        # Fetch only those gt_maps
        gt_label = []
        count = []
        for gt_map in gt_maps:
            coords, count_curr = fetch_coords_from_map(gt_map)
            coords = coords.to(gpu_num)
            gt_label.append(coords)
            count.append(count_curr)
        count = torch.tensor(count).to(gpu_num)
        imgs = imgs.view(-1, 3, 224, 224).to(gpu_num)
        # traj_coords = torch.ones((imgs.shape[0], 2), dtype = int) * -1

        # Forward Prop
        out, pred_count, extracted_features = model(imgs)
        pred_count = pred_count.view(-1)
        out = out.squeeze(1)
        orig_sizes = torch.ones((out.shape[0],2), device=imgs.device)*224
        loss = loss_weighted_hausdroff(out, gt_label, orig_sizes, pred_count, count)
        
        # # Frames where there is no keypoint or the frame of video does not exist so everything is 0
        # # And hence no gradient can be calculated
        if not loss.requires_grad:
            continue
        loss.backward()
        optim.step()
        # torch.cuda.synchronize()
        loop.set_postfix(train_loss = loss.item())
        epoch_loss += loss.item()
    epoch_loss /= len(train_loader)
    loop.set_postfix(train_loss = epoch_loss)

    model.eval()
    loop = tqdm(enumerate(valid_loader), total= len(valid_loader), leave=False) 
    loop.set_description(f"Valid Epoch [{epoch+1}/{epochs}]")
    valid_epoch_loss = 0.0
    with torch.no_grad():
        for (idx, batch) in loop:
            optim.zero_grad()
            imgs, gt_maps = batch
            gt_maps = gt_maps.view(-1, 224, 224)

            # Fetch only those gt_maps
            gt_label = []
            count = []
            for gt_map in gt_maps:
                coords, count_curr = fetch_coords_from_map(gt_map)
                coords = coords.to(gpu_num)
                gt_label.append(coords)
                count.append(count_curr)
            count = torch.tensor(count).to(gpu_num)
            imgs = imgs.view(-1, 3, 224, 224).to(gpu_num)

            # Forward Prop
            out, pred_count, extracted_features = model(imgs)
            pred_count = pred_count.view(-1)
            out = out.squeeze(1)
            orig_sizes = torch.ones((out.shape[0],2), device=imgs.device)*224
            loss = loss_weighted_hausdroff(out, gt_label, orig_sizes, pred_count, count)
            loop.set_postfix(valid_loss = loss.item())
            valid_epoch_loss += loss.item()
    valid_epoch_loss /= len(valid_loader)
    loop.set_postfix(valid_loss = valid_epoch_loss)

    scheduler.step(valid_epoch_loss)
    print(f'###### {epoch} ######')
    print(f'TRAIN LOSS: {epoch_loss:.3f}, VALID LOSS: {valid_epoch_loss:.3f}')
    if valid_epoch_loss < best_valid_loss:
        best_valid_loss = valid_epoch_loss
        print('Saving Model')
        torch.save(model.state_dict(), pretrain_emb_path)

print(f"Keypoint Embeddings are saved at: {pretrain_emb_path}")