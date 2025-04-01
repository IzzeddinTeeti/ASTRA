import torch
from torch import nn
import math

class ConditionalVariationalEncoder(nn.Module):
    def __init__(self, cfg):
        super(ConditionalVariationalEncoder, self).__init__()
        self.cfg = cfg

        self.device = self.cfg.device
        self.num_device = len(self.cfg.device_list)
        self.batch_size = self.cfg.TRAIN.BATCH_SIZE
        self.batch_size_device = int(self.batch_size // self.num_device) if self.num_device > 0 else self.batch_size
        if self.cfg.DATASET == 'ETH_UCY':
            # if self.cfg.SUBSET == 'univ':
            #     self.num_pedestrians = 2
            # elif self.cfg.SUBSET == 'zara01':
            #     self.num_pedestrians = 3
            # else:
            self.num_pedestrians = 1
        else:
            self.num_pedestrians = 1

        self.obs_len = int(self.cfg.PREDICTION.OBS_TIME * self.cfg.DATA.FREQUENCY)
        self.pred_len = math.ceil(self.cfg.PREDICTION.PRED_TIME * self.cfg.DATA.FREQUENCY)
        self.input_dim = 2 if self.cfg.DATASET == 'ETH_UCY' else 4
        self.unet_dim = self.cfg.MODEL.UNET_DIM if self.cfg.MODEL.USE_PRETRAINED_UNET else 0
        self.spatial_dim = self.cfg.MODEL.SPATIAL_DIM
        self.velo_dim = self.cfg.MODEL.VELO_DIM if self.cfg.MODEL.INC_VELO else 0
        self.walk_length = self.cfg.MODEL.RAND_WALK_LEN if self.cfg.MODEL.USE_SOCIAL else 0
        self.temp_dim = self.cfg.MODEL.TEMP_DIM
        self.scene_dim = self.unet_dim + self.temp_dim if self.cfg.MODEL.USE_PRETRAINED_UNET else 0
        self.token_dim = self.spatial_dim + self.temp_dim + self.velo_dim + self.walk_length
        self.latent_dim = self.cfg.MODEL.LATENT_DIM
        self.num_samples = 1
        self.cvae_dim = self.obs_len*(self.spatial_dim + self.temp_dim) + self.pred_len*(self.spatial_dim + self.temp_dim)
        self.cvae_encoder = nn.Sequential(
            nn.Linear(self.cvae_dim, self.latent_dim*2),
            nn.LeakyReLU(0.2),
            nn.Linear(self.latent_dim*2, self.latent_dim),
            nn.LeakyReLU(0.2),
        )
        
        self.mu_c = nn.Linear(self.latent_dim, self.latent_dim)
        self.log_variance_c = nn.Linear(self.latent_dim, self.latent_dim)

        self.cvae_decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim*2),
            nn.LeakyReLU(0.2),
            nn.Linear(self.latent_dim*2, self.obs_len * (self.token_dim + self.scene_dim)),
            nn.LeakyReLU(0.2),
        )
        
        self.cvae_decoder.apply(self.init_weights)
        self.cvae_encoder.apply(self.init_weights)
        self.log_variance_c.apply(self.init_weights)
        self.mu_c.apply(self.init_weights)

    def init_weights(self, layer):
        if isinstance(layer, nn.Linear):
            torch.nn.init.kaiming_normal_(layer.weight)

    def forward(self, x_encoded, y_encoded):
        encoded = torch.cat((x_encoded, y_encoded), dim = -1)
        encoded = self.cvae_encoder(encoded)
        mean = torch.unsqueeze(self.mu_c(encoded), dim = -2)
        log_var = torch.unsqueeze(self.log_variance_c(encoded), dim = -2)
        var = torch.exp(0.5*log_var)
        with torch.no_grad():
            samples = torch.randn(self.batch_size, self.num_pedestrians, self.num_samples, self.latent_dim, device = self.cfg.device)
        output = var * samples + mean
        decoded_output = self.cvae_decoder(output) 

        return mean, log_var, decoded_output

class VariationalEncoder(nn.Module):

    def __init__(self, cfg):
        super(VariationalEncoder, self).__init__()
        self.cfg = cfg

        self.device = self.cfg.device
        self.num_device = len(self.cfg.device_list)
        self.batch_size = self.cfg.TRAIN.BATCH_SIZE
        self.batch_size_device = int(self.batch_size // self.num_device) if self.num_device > 0 else self.batch_size
        # if self.cfg.DATASET == 'ETH_UCY':
        #     if self.cfg.SUBSET == 'univ':
        #         self.num_pedestrians = 2
        #     elif self.cfg.SUBSET == 'zara01':
        #         self.num_pedestrians = 3
        #     else:
        #         self.num_pedestrians = 1
        # else:
        self.num_pedestrians = 1

        self.obs_len = int(self.cfg.PREDICTION.OBS_TIME * self.cfg.DATA.FREQUENCY)
        self.input_dim = 2 if self.cfg.DATASET == 'ETH_UCY' else 4
        self.unet_dim = self.cfg.MODEL.UNET_DIM if self.cfg.MODEL.USE_PRETRAINED_UNET else 0
        self.spatial_dim = self.cfg.MODEL.SPATIAL_DIM
        self.velo_dim = self.cfg.MODEL.VELO_DIM if self.cfg.MODEL.INC_VELO else 0
        self.walk_length = self.cfg.MODEL.RAND_WALK_LEN if self.cfg.MODEL.USE_SOCIAL else 0
        self.temp_dim = self.cfg.MODEL.TEMP_DIM
        self.scene_dim = self.unet_dim + self.temp_dim if self.cfg.MODEL.USE_PRETRAINED_UNET else 0
        self.token_dim = self.spatial_dim + self.temp_dim + self.velo_dim + self.walk_length
        self.latent_dim = self.cfg.MODEL.LATENT_DIM
        self.num_samples = self.cfg.MODEL.K 

        self.vae_encoder = nn.Sequential(
            nn.Linear(self.obs_len * (self.token_dim + self.scene_dim), self.latent_dim*2),
            nn.LeakyReLU(0.2),
            nn.Linear(self.latent_dim*2, self.latent_dim),
            nn.LeakyReLU(0.2),
        )
        
        self.mu = nn.Linear(self.latent_dim, self.latent_dim)
        self.log_variance = nn.Linear(self.latent_dim, self.latent_dim)

        self.vae_decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim*2),
            nn.LeakyReLU(0.2),
            nn.Linear(self.latent_dim*2, self.obs_len * (self.token_dim + self.scene_dim)),
            nn.LeakyReLU(0.2),
        )
        
        self.vae_decoder.apply(self.init_weights)
        self.vae_encoder.apply(self.init_weights)
        self.log_variance.apply(self.init_weights)
        self.mu.apply(self.init_weights)

    def init_weights(self, layer):
        if isinstance(layer, nn.Linear):
            torch.nn.init.kaiming_normal_(layer.weight)

    def forward(self, x):
        encoded = self.vae_encoder(x)
        mean = torch.unsqueeze(self.mu(encoded), dim = -2)
        log_var = torch.unsqueeze(self.log_variance(encoded), dim = -2)
        var = torch.exp(0.5*log_var)
        with torch.no_grad():
            samples = torch.randn(self.batch_size, self.num_pedestrians, self.num_samples, self.latent_dim, device = self.cfg.device)
        output = var * samples + mean
        decoded_output = self.vae_decoder(output) 

        return mean, log_var, decoded_output