import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T 
from torch_geometric.data import Data
import math
import warnings
from models.vae import VariationalEncoder
from models.vae import ConditionalVariationalEncoder
warnings.filterwarnings('ignore')
from icecream import ic

class ASTRA_model(nn.Module):
    def __init__(self, cfg):
        super(ASTRA_model, self).__init__()

        self.cfg = cfg
        self.device = self.cfg.device
        self.num_device = len(self.cfg.device_list)
        self.batch_size = self.cfg.TRAIN.BATCH_SIZE
        self.batch_size_device = int(self.batch_size // self.num_device) if self.num_device > 0 else self.batch_size

        self.obs_len = int(self.cfg.PREDICTION.OBS_TIME * self.cfg.DATA.FREQUENCY)
        self.pred_len = math.ceil(self.cfg.PREDICTION.PRED_TIME * self.cfg.DATA.FREQUENCY)

        # Set the number of pedestrians and the edge index
        if self.cfg.DATASET == 'ETH_UCY':
            if self.cfg.SUBSET == 'univ':
                self.num_pedestrians = 2
                self.edge_index = torch.tensor([[0, 1], 
                                                [1, 0]], dtype=torch.long)
            elif self.cfg.SUBSET == 'zara01':
                self.num_pedestrians = 3
                self.edge_index = torch.tensor([[0, 0, 1, 1, 2, 2], 
                                                [1, 2, 0, 2, 0, 1]], dtype=torch.long)
            else:
                self.num_pedestrians = 1
                self.edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            self.num_pedestrians = 1
            self.edge_index = torch.empty((2, 0), dtype=torch.long)
        
        # Dimensions of different encodings
        self.input_dim = 2 if self.cfg.DATASET == 'ETH_UCY' else 4
        self.output_dim = 2 if self.cfg.DATASET == 'ETH_UCY' else 4
        self.unet_dim = self.cfg.MODEL.UNET_DIM if self.cfg.MODEL.USE_PRETRAINED_UNET else 0
        self.spatial_dim = self.cfg.MODEL.SPATIAL_DIM
        self.velo_dim = self.cfg.MODEL.VELO_DIM if self.cfg.MODEL.INC_VELO else 0
        self.walk_length = self.cfg.MODEL.RAND_WALK_LEN if self.cfg.MODEL.USE_SOCIAL else 0
        self.temp_dim = self.cfg.MODEL.TEMP_DIM
        self.scene_dim = self.unet_dim + self.temp_dim if self.cfg.MODEL.USE_PRETRAINED_UNET else 0
        self.output_dim = 2 if self.cfg.DATASET == 'ETH_UCY' else 4
        self.edge_criteria = self.cfg.MODEL.EDGE_CRITERIA
        if self.cfg.MODEL.USE_VAE:
            self.latent_dim = self.cfg.MODEL.LATENT_DIM
            self.num_samples = self.cfg.MODEL.K       
            self.vae_model = VariationalEncoder(self.cfg) 
            self.cvae_model = ConditionalVariationalEncoder(self.cfg)
                
        # Linear Projection: XY coordinates
        self.lin_proj_xy = nn.Sequential(
            nn.Linear(in_features=self.input_dim, out_features=self.spatial_dim),
        )
        self.lin_proj_xy.apply(self.init_weights)
        
        # Linear Projection: UNET features
        if self.cfg.MODEL.USE_PRETRAINED_UNET:
            self.lin_proj_unet = nn.Sequential(
                nn.Linear(in_features=self.cfg.MODEL.FEATURE_DIM, out_features=self.unet_dim),
            )
            self.lin_proj_unet.apply(self.init_weights)   
        
        # Linear Projection: Velocity        
        if self.cfg.MODEL.INC_VELO:
            self.lin_proj_velo = nn.Sequential(
                nn.Linear(in_features=1, out_features=self.velo_dim),
            )
            self.lin_proj_velo.apply(self.init_weights)
        
        # Linear Projection: Social Encodings
        if self.cfg.MODEL.USE_SOCIAL:
            self.lin_proj_social = nn.Sequential(
                nn.Linear(in_features=self.walk_length, out_features=self.walk_length),
            )
            self.lin_proj_social.apply(self.init_weights)        
        
        # Agent Aware Transformer Encoder
        self.token_dim = self.spatial_dim + self.temp_dim + self.velo_dim + self.walk_length
        self.norm = nn.LayerNorm(self.token_dim, eps = 1e-6)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.token_dim, nhead=self.cfg.MODEL.NHEAD, dim_feedforward=self.cfg.MODEL.DIM_FEEDFORWARD, dropout=self.cfg.MODEL.DROPOUT, activation='gelu', batch_first=True)
        self.agent_aware_trans_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.cfg.MODEL.ENC_LAYERS, norm = self.norm)

        # Scene Aware Transformer Encoder
        if self.cfg.MODEL.USE_PRETRAINED_UNET:
            self.unet_trans_norm = nn.LayerNorm(self.unet_dim+self.temp_dim, eps = 1e-6)
            scene_encoder_layer = nn.TransformerEncoderLayer(d_model=self.unet_dim+self.temp_dim, nhead=self.cfg.MODEL.NHEAD, dim_feedforward=self.cfg.MODEL.DIM_FEEDFORWARD, dropout=self.cfg.MODEL.DROPOUT, activation='gelu', batch_first=True)
            self.scene_aware_trans_encoder = nn.TransformerEncoder(scene_encoder_layer, num_layers=self.cfg.MODEL.ENC_LAYERS, norm = self.unet_trans_norm)
         
        # Transformer Masking
        if self.cfg.MODEL.TRANS_MASK:
            self.mask = self.create_custom_mask()
        else:
            self.mask = None
        
        # MLP Decoder
        self.decoder = nn.Sequential(
            nn.Linear(in_features = self.obs_len * (self.token_dim + self.scene_dim), 
                      out_features = self.pred_len * self.output_dim),
        )
        self.decoder.apply(self.init_weights)
   
    def init_weights(self, layer):
        if isinstance(layer, nn.Linear):
            torch.nn.init.xavier_normal_(layer.weight)

    def __len__(self):
        pass
    
    def scene_encoder(self, unet_features, temporal_encodings):
        unet_proj = self.lin_proj_unet(unet_features)
        # ic(unet_proj)
        ic(unet_proj.shape)                  # unet_output: (Batch, Frames, Agents, unet_dim)
        unet_trans_input = torch.cat([unet_proj, temporal_encodings], dim=-1)
        unet_trans_input = unet_trans_input.view(self.batch_size_device, -1, self.scene_dim)
        # ic(unet_trans_input)
        ic(unet_trans_input.shape)           # unet_trans_input: (Batch, Frames, Agents, unet_dim)
        scene_encodings = self.scene_aware_trans_encoder(unet_trans_input, mask = self.mask)
        # ic(scene_encodings)
        ic(scene_encodings.shape)             # scene_encodings: (Batch, Frames, Agents, unet_dim+temp_dim)
        return scene_encodings
        
    def xy_encoder(self, xy_coords):
        spatial_output = self.lin_proj_xy(xy_coords)
        return spatial_output
    
    def velo_encoder(self, past_loc):
        differences = past_loc[:, :, 1:, :] - past_loc[:, :, :-1, :] 
        velocities = torch.sqrt((differences ** 2).sum(-1) + 1e-6)
        first_frame_velo = torch.zeros(past_loc.shape[0], past_loc.shape[1], 1, device=past_loc.device)
        velocities = torch.cat([first_frame_velo, velocities], dim=2)
        velocities = velocities.permute(0, 2, 1).unsqueeze(-1)
        # ic(velocities)
        ic(velocities.shape)                  # velocities: (Batch, Frames, Agents, 1)
        velo_output = self.lin_proj_velo(velocities)
        return velo_output
    
    def create_distance_adjacency_matrix(self, past_loc):
        adjacency_matrix = torch.cdist(past_loc, past_loc, p=2.0)
        adjacency_matrix = torch.reciprocal(adjacency_matrix) 
        identity_matrix = torch.eye(self.num_pedestrians).bool().unsqueeze(0).unsqueeze(1).to(adjacency_matrix.device)
        mask = ~identity_matrix
        masked_tensor = adjacency_matrix.masked_select(mask.unsqueeze(0))
        adjacency_matrix = masked_tensor.view(self.batch_size_device, self.obs_len, self.num_pedestrians * (self.num_pedestrians - 1))
        adjacency_matrix[adjacency_matrix == float('inf')] = 0 # replace inf with 0
        return adjacency_matrix

    def social_encoder(self, past_loc):
        nodes = past_loc
        self.edge_index = self.edge_index.to(past_loc.device)
        if self.edge_criteria == 'distance':
            adjacency_matrix = self.create_distance_adjacency_matrix(nodes) 
        social_encodings = torch.zeros((nodes.shape[0], nodes.shape[1], nodes.shape[2], self.walk_length), device=nodes.device)
        for b in range(nodes.shape[0]):
            for f in range(nodes.shape[1]):
                data = Data(x=nodes[b, f], edge_index=self.edge_index, edge_weight=adjacency_matrix[b, f], device=nodes.device)
                random_walk_pe = T.AddRandomWalkPE(walk_length=self.walk_length)
                data_t = random_walk_pe(data)
                social_encodings[b, f] = data_t.random_walk_pe
        social_output = self.lin_proj_social(social_encodings)
        return social_output

    def temporal_encoder(self):
        position = torch.arange(0, self.obs_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.temp_dim, 2).float() * (-math.log(10000.0) / self.temp_dim))

        # Calculate sine and cosine components of positional encoding
        pos_enc = torch.zeros(self.obs_len, self.temp_dim)
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)

        pos_enc = pos_enc.unsqueeze(0).expand(self.batch_size_device, -1, -1)
        pos_enc = pos_enc.unsqueeze(2).expand(-1, -1, self.num_pedestrians, -1).to(self.device)
        return pos_enc
    
    def future_temporal_encoder(self):
        position = torch.arange(self.obs_len, self.obs_len + self.pred_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.temp_dim, 2).float() * (-math.log(10000.0) / self.temp_dim))

        # Calculate sine and cosine components of positional encoding
        pos_enc = torch.zeros(self.pred_len, self.temp_dim)
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)

        pos_enc = pos_enc.unsqueeze(0).expand(self.batch_size_device, -1, -1)
        pos_enc = pos_enc.unsqueeze(2).expand(-1, -1, self.num_pedestrians, -1).to(self.device)
        return pos_enc

    def aggregate(self, spatial_output, velo_output, social_output, temporal_output):        
        outputs = []
        outputs.append(spatial_output)
        outputs.append(temporal_output)
        if self.cfg.MODEL.INC_VELO:
            outputs.append(velo_output)
        if self.cfg.MODEL.USE_SOCIAL:
            outputs.append(social_output)
        output = torch.cat(outputs, dim=-1)                    
        output = output.view(self.batch_size_device, -1, self.token_dim)
        return output

    def create_custom_mask(self):
        mask = torch.full((self.obs_len * self.num_pedestrians, self.obs_len * self.num_pedestrians), float('-inf'))
        for frame in range(self.obs_len):
            for pedestrian in range(self.num_pedestrians):
                index = frame * self.num_pedestrians + pedestrian
                mask[index, pedestrian::self.num_pedestrians] = 0.0
        return mask.to(self.device)
    
    def forward(self, past_loc, fut_loc, unet_features, mode = 'train'):

        # Spatial Encodings (x, y)
        spatial_features = past_loc.permute(0, 2, 1, 3) 
        # ic(spatial_features)
        ic(spatial_features.shape)           # spatial_features: (Batch, Frames, Agents, 2)
        spatial_encodings = self.xy_encoder(spatial_features)
        # ic(spatial_encodings)
        ic(spatial_encodings.shape)          # spatial_encodings: (Batch, Frames, Agents, spatial_dim)
        
        # Velocity Encodings
        if self.cfg.MODEL.INC_VELO:
            velo_encodings = self.velo_encoder(past_loc)
            # ic(velo_encodings)
            ic(velo_encodings.shape)         # velo_encodings: (Batch, Frames, Agents, velo_dim)
        else:
            velo_encodings = None 
        
        # Temporal Encodings
        temporal_encodings = self.temporal_encoder()

        if mode == 'train' and self.cfg.MODEL.USE_VAE:
            spatial_features_y = fut_loc.permute(0, 2, 1, 3)
            spatial_encodings_y = self.xy_encoder(spatial_features_y)
            temporal_encodings_y = self.future_temporal_encoder()
            ic(spatial_encodings_y.shape)
            ic(temporal_encodings_y.shape)
            encodings_y = torch.cat([spatial_encodings_y, temporal_encodings_y], dim=-1)
            encodings_x = torch.cat([spatial_encodings, temporal_encodings], dim=-1)
            encodings_y = encodings_y.view(self.batch_size_device, self.num_pedestrians, -1)
            encodings_x = encodings_x.view(self.batch_size_device, self.num_pedestrians, -1)
            ic(encodings_x.shape, encodings_y.shape)
            mean_c, log_var_c, MLP_input_c = self.cvae_model(encodings_x, encodings_y)
            ic(mean_c.shape)
            ic(log_var_c.shape)
            ic(MLP_input_c.shape)
        else:
            mean_c, log_var_c, MLP_input_c = None, None, None

        # ic(temporal_encodings)
        ic(temporal_encodings.shape)        # temporal_encodings: (Batch, Frames, Agents, temp_dim)
        
        # Scene Encodings
        if self.cfg.MODEL.USE_PRETRAINED_UNET:
            unet_features = unet_features.permute(0, 2, 1, 3)
            # ic(unet_features)
            ic(unet_features.shape)          # unet_features: (Batch, Frames, Agents, feature_dim)
            scene_encodings = self.scene_encoder(unet_features, temporal_encodings)
        else:
            scene_encodings = None
            
        # Social Encodings
        if self.cfg.MODEL.USE_SOCIAL:
            social_encodings = self.social_encoder(spatial_features)
            # ic(social_encodings)
            ic(social_encodings.shape)             # social_encodings: (Batch, Frames, Agents, walk_length)        
        else:
            social_encodings = None
                
        # Agent Aware Transformer Encoder 
        trans_input = self.aggregate(spatial_encodings, velo_encodings, social_encodings, temporal_encodings)   
        # ic(trans_input)     
        ic(trans_input.shape)                      # trans_input: (Batch, Frames*Agents, token_dim)
        if self.cfg.MODEL.TRANS_MASK:
            ic(self.mask)
            ic(self.mask.shape)                    # mask: (Frames*Agents, Frames*Agents)
        trans_encodings = self.agent_aware_trans_encoder(trans_input, mask = self.mask)
        # ic(trans_encodings)
        ic(trans_encodings.shape)                  # trans_encodings: (Batch, Frames*Agents, token_dim)                             
        
        # MLP Inputs
        MLP_inputs = [] 
        MLP_inputs.append(trans_encodings)
        if self.cfg.MODEL.USE_PRETRAINED_UNET:
            MLP_inputs.append(scene_encodings)
        MLP_input = torch.cat(MLP_inputs, dim=-1)
        # ic(MLP_input)
        ic(MLP_input.shape)                               # MLP_input: (Batch, Frames*Agents, token_dim + unet_dim

        # if self.cfg.MODEL.USE_VAE:
        #     ic(spatial_features.shape, spatial_features)
        #     past_loc_c = spatial_features.view(self.batch_size_device,  -1, 2)
        #     ic(MLP_input.shape, MLP_input)
        #     MLP_input = torch.cat([MLP_input, past_loc_c], dim = -1)
        #     ic(MLP_input.shape, MLP_input)
        
        MLP_input = MLP_input.view(self.batch_size_device, self.num_pedestrians, -1)
        # ic(MLP_input)
        ic(MLP_input.shape)                               # MLP_input: (Batch, Agents, Frames*(token_dim + unet_dim + temp_dim))
        if self.cfg.MODEL.USE_VAE:
            mean, log_var, MLP_input = self.vae_model(MLP_input)
            
            ic(MLP_input.shape)
        else:
            mean, log_var = None, None

        # MLP Decoder
        MLP_decodings = self.decoder(MLP_input)
        # ic(MLP_decodings) 
        ic(MLP_decodings.shape)                           # MLP_decodings: (Batch, Agents, Pred_len*output_dim)
        
        # Model Output
        if self.cfg.MODEL.USE_VAE:
            model_output = MLP_decodings.view(self.batch_size_device, self.num_pedestrians, self.num_samples, self.pred_len, self.output_dim)
        else:
            model_output = MLP_decodings.view(self.batch_size_device, self.num_pedestrians, self.pred_len, self.output_dim)
        
        # ic(model_output)
        ic(model_output.shape)                            # model_output: (Batch, Agents, K, Pred_len, output_dim)
        if mode == 'train' and self.cfg.MODEL.USE_VAE:
            c_mlp_decodings = self.decoder(MLP_input_c)
            c_model_output = c_mlp_decodings.view(self.batch_size_device, self.num_pedestrians, 1, self.pred_len, self.output_dim)
            return mean, log_var, model_output, mean_c, log_var_c, c_model_output
        return mean, log_var, model_output, None, None, None