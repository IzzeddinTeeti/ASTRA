import segmentation_models_pytorch as smp
import torch
from torch import nn

class UNETEmbeddingExtractor(nn.Module):
    def __init__(self, cfg):
        super(UNETEmbeddingExtractor, self).__init__()
        self.unet = smp.Unet(cfg.MODEL.FEATURE_EXTRACTOR, classes=1, activation='sigmoid')
        in_channels_extractor = self.unet.segmentation_head[0].in_channels
        self.feature_extractor = nn.Sequential(nn.Conv2d(in_channels=in_channels_extractor, out_channels=128, kernel_size=1),
                                               nn.BatchNorm2d(128),
                                                nn.ReLU())
        self.feature_extractor.apply(self.init_weights)
        self.seg_head = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1),
                                    nn.Sigmoid())
        self.seg_head.apply(self.init_weights)
        self.unet.segmentation_head = nn.Identity()

        self.flatten_layer = nn.Flatten()
        if cfg.MODEL.FEATURE_EXTRACTOR == 'resnet50':
            flatten_dim = 2048
        if cfg.MODEL.FEATURE_EXTRACTOR == 'resnet34' or cfg.MODEL.FEATURE_EXTRACTOR == 'resnet18':
            flatten_dim = 512
        self.branch1 = nn.Sequential(nn.Linear(flatten_dim, 64), # Change flatten_dim to corresponding dimension after flattening
                                     nn.BatchNorm1d(64),
                                     nn.ReLU(inplace=True),
                                     nn.Dropout(p = 0.5))
        self.branch2 = nn.Sequential(nn.Linear(224*224, 64),
                                     nn.BatchNorm1d(64),
                                     nn.ReLU(inplace=True),
                                     nn.Dropout(p = 0.5))
        self.regression_head = nn.Sequential(nn.Linear(128, 1),
                                             nn.ReLU(inplace=True))
        self.global_pool = nn.AdaptiveMaxPool2d(1)
        self.pretrain_flag = cfg.UNET_MODE

    def init_weights(self, layer):
        if isinstance(layer, nn.Conv2d):
            torch.nn.init.xavier_normal_(layer.weight)
            # layer.bias.data.fill_(0.0)

    def forward(self, x):
        if self.pretrain_flag == 'training':
            features = self.feature_extractor(self.unet(x))
            bottleneck_features = self.unet.encoder(x)[-1]
            out = self.seg_head(features)

            # Number of object
            out_flat = self.flatten_layer(out)
            extracted_features = self.flatten_layer(self.global_pool(bottleneck_features))
            bottleneck_proj = self.branch1(extracted_features)
            out_proj = self.branch2(out_flat)
            regression_feat = torch.cat((bottleneck_proj, out_proj), dim = 1)
            n_out = self.regression_head(regression_feat)

            # Filter out coordinates with (-1, -1)
            # valid_coords_mask = torch.all(pixel_coords != -1, axis = 1)

            # # Fetching Keypoint Extractors
            # batch_indices = torch.arange(features.shape[0])
            # extracted_features = features[batch_indices, :, pixel_coords[:, 0], pixel_coords[:, 1]]
            # extracted_features[~valid_coords_mask] = 0.

            return out, n_out, extracted_features
        else:
            bottleneck_features = self.unet.encoder(x)[-1]
            extracted_features = self.flatten_layer(self.global_pool(bottleneck_features))
            return None, None, extracted_features