import icecream as ic
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
import math
from sklearn.utils.extmath import cartesian
from einops import rearrange

class Loss(nn.Module):
    def __init__(self, cfg):
        super(Loss, self).__init__()
        self.loss_func = cfg.LOSS.FUN
        self.device = cfg.device
        self.pred_len = math.ceil(cfg.PREDICTION.PRED_TIME * cfg.DATA.FREQUENCY)
        
        if cfg.LOSS.WEIGHTED_PENALTY:
            if cfg.LOSS.WEIGHTED_PENALTY == 'linear':
                self.weights = torch.linspace(cfg.LOSS.START_WEIGHT, cfg.LOSS.END_WEIGHT, steps=self.pred_len, device=self.device, requires_grad=False)
            elif cfg.LOSS.WEIGHTED_PENALTY == 'quadratic':
                self.weights = torch.linspace(cfg.LOSS.START_WEIGHT, cfg.LOSS.END_WEIGHT, steps=self.pred_len, device=self.device, requires_grad=False) ** 2
            elif cfg.LOSS.WEIGHTED_PENALTY == 'exponential':
                self.weights = torch.exp(torch.linspace(math.log(cfg.LOSS.START_WEIGHT), math.log(cfg.LOSS.END_WEIGHT), steps=self.pred_len, device=self.device, requires_grad=False))
            elif cfg.LOSS.WEIGHTED_PENALTY == 'parabolic':
                x = torch.linspace(-1, 1, steps=self.pred_len, device=self.device)
                self.weights = (cfg.LOSS.MAX_WEIGHT - cfg.LOSS.MIN_WEIGHT) * x.pow(2) + cfg.LOSS.MIN_WEIGHT
            
        if self.loss_func == 'MSE' or self.loss_func == 'RMSE':
            self.criterion = nn.MSELoss(reduction='none')
            
        elif self.loss_func == 'SmoothL1':
            self.criterion = nn.SmoothL1Loss(reduction='none')

        self.samples_point_weight = torch.linspace(1, 0, steps = self.pred_len, device=self.device, requires_grad=False)

    def forward(self, pred, target, cfg, k_pred = False):
        target = torch.unsqueeze(target, dim = -3)
        loss = self.criterion(pred, target)
        loss = loss[...,0] + loss[...,1]
        if self.loss_func == 'RMSE':
            loss = torch.sqrt(loss + 1e-06)
            
        if cfg.LOSS.WEIGHTED_PENALTY:
            loss = loss * self.weights.view(1, 1, 1, -1)
        if k_pred:
            loss = loss.mean(dim = -1)
            loss = loss.min()
        else:
            loss = loss.mean()
        return loss

class KLDivergenceLoss(nn.Module):
    def __init__(self):
        super(KLDivergenceLoss, self).__init__()

    def forward(self, mean, log_var):
        return -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    
class GaussianKLDivergenceLoss(nn.Module):
    def __init__(self):
        super(GaussianKLDivergenceLoss, self).__init__()

    def forward(self, mu_p, logvar_p, mu_q, logvar_q):
        mu_p = rearrange(mu_p, 'b a k l -> (b a k) l')
        logvar_p = rearrange(logvar_p, 'b a k l -> (b a k) l')
        mu_q = rearrange(mu_q, 'b a k l -> (b a k) l')
        logvar_q = rearrange(logvar_q, 'b a k l -> (b a k) l')

        sigma_p_sq = torch.exp(logvar_p)
        sigma_q_sq = torch.exp(logvar_q)
        # Calculate the KL divergence (By Chatgpt)
        # kl_div = 0.5 * torch.sum(
        #     (sigma_p_sq / sigma_q_sq) +
        #     ((mu_q - mu_p)**2 / sigma_q_sq) -
        #     1 + logvar_q - logvar_p
        # )

        # From Bitrap
        kl_div = 0.5 * ((sigma_q_sq/sigma_p_sq) +
                        ((mu_p - mu_q)**2/sigma_p_sq) - 
                        1 + (logvar_p - logvar_q))
        kl_div = kl_div.sum(dim=-1).mean()
        return kl_div
        
class DiversityLoss(nn.Module):
    def __init__(self):
        super(DiversityLoss, self).__init__()

    # def forward(self, pred_traj, gpu_num):
    #     print(self.m(pred_traj, gpu_num))
    #     inp_rearrange = rearrange(pred_traj, 'b a k f c -> (b a) k f c')
    #     pairwise_batch_differences = torch.zeros((len(inp_rearrange)))
    #     for i in range(len(inp_rearrange)):
    #         pairwise_diff = torch.zeros((1, inp_rearrange.shape[-3], inp_rearrange.shape[-3]))
    #         for k1 in range(len(inp_rearrange[i])):
    #             for k2 in range(len(inp_rearrange[i])):
    #                 if(k1 != k2):
    #                     diff = torch.exp(((inp_rearrange[i][k1] - inp_rearrange[i][k2])**2).sum(dim = -1).mean()/10)
    #                     pairwise_diff[0][k1][k2] = diff
    #         # print(pairwise_diff)
    #         pairwise_diff = pairwise_diff.mean()
    #         pairwise_batch_differences[i] = pairwise_diff
    #     return pairwise_batch_differences.mean()
    
    def forward(self,pred_traj, gpu_num):
        inp_rearrange = rearrange(pred_traj, 'b a k f c -> (b a) k f c')

        # Calculate pairwise differences using vectorized operations
        pairwise_diff = torch.exp(((inp_rearrange.unsqueeze(2) - inp_rearrange.unsqueeze(1))**2).sum(dim=-1).mean(dim=-1) / 10)
        # Set diagonal elements to zero
        mask = ~torch.eye(inp_rearrange.size(1), device=gpu_num).bool()
        pairwise_diff = pairwise_diff * mask.unsqueeze(0)
        
        # Compute the mean over the last two dimensions
        pairwise_diff = pairwise_diff.mean(dim=(-2, -1))

        # Compute the mean over the batch
        loss = pairwise_diff.mean()
        return loss

# Loss Function
# Ref: https://github.com/javiribera/locating-objects-without-bboxes/tree/master
# Paper Ref: https://openaccess.thecvf.com/content_CVPR_2019/papers/Ribera_Locating_Objects_Without_Bounding_Boxes_CVPR_2019_paper.pdf
class WeightedHausdorffDistance(nn.Module):
    def __init__(self,
                 resized_height, resized_width,
                 p=-9,
                 return_2_terms=False,
                 device=torch.device('cpu')):
        """
        :param resized_height: Number of rows in the image.
        :param resized_width: Number of columns in the image.
        :param p: Exponent in the generalized mean. -inf makes it the minimum.
        :param return_2_terms: Whether to return the 2 terms
                               of the WHD instead of their sum.
                               Default: False.
        :param device: Device where all Tensors will reside.
        """
        super(WeightedHausdorffDistance, self).__init__()

        # Prepare all possible (row, col) locations in the image
        self.height, self.width = resized_height, resized_width
        self.resized_size = torch.tensor([resized_height,
                                          resized_width],
                                         dtype=torch.get_default_dtype(),
                                         device=device)
        self.max_dist = math.sqrt(resized_height**2 + resized_width**2)
        self.n_pixels = resized_height * resized_width
        self.all_img_locations = torch.from_numpy(cartesian([np.arange(resized_height),
                                                             np.arange(resized_width)]))
        # Convert to appropiate type
        self.all_img_locations = self.all_img_locations.to(device=device,
                                                           dtype=torch.get_default_dtype())

        self.return_2_terms = return_2_terms
        self.p = p
        self.device = device
        self.smooth_l1 = nn.SmoothL1Loss()

    def forward(self, prob_map, gt, orig_sizes, pred_count, true_count):
        """
        Compute the Weighted Hausdorff Distance function
        between the estimated probability map and ground truth points.
        The output is the WHD averaged through all the batch.

        :param prob_map: (B x H x W) Tensor of the probability map of the estimation.
                         B is batch size, H is height and W is width.
                         Values must be between 0 and 1.
        :param gt: List of Tensors of the Ground Truth points.
                   Must be of size B as in prob_map.
                   Each element in the list must be a 2D Tensor,
                   where each row is the (y, x), i.e, (row, col) of a GT point.
        :param orig_sizes: Bx2 Tensor containing the size
                           of the original images.
                           B is batch size.
                           The size must be in (height, width) format.
        :return: Single-scalar Tensor with the Weighted Hausdorff Distance.
                 If self.return_2_terms=True, then return a tuple containing
                 the two terms of the Weighted Hausdorff Distance.
        """

        self._assert_no_grad(gt)

        assert prob_map.dim() == 3, 'The probability map must be (B x H x W)'
        assert prob_map.size()[1:3] == (self.height, self.width), \
            'You must configure the WeightedHausdorffDistance with the height and width of the ' \
            'probability map that you are using, got a probability map of size %s'\
            % str(prob_map.size())

        batch_size = prob_map.shape[0]
        assert batch_size == len(gt)

        terms_1 = []
        terms_2 = []
        for b in range(batch_size):
            # One by one
            prob_map_b = prob_map[b, :, :]
            gt_b = gt[b]
            orig_size_b = orig_sizes[b, :]
            norm_factor = (orig_size_b/self.resized_size).unsqueeze(0)
            n_gt_pts = gt_b.size()[0]

            # Corner case: no GT points
            if gt_b.ndimension() == 1 and (gt_b < 0).all().item() == 1:
                terms_1.append(torch.tensor(0, device=self.device,
                                            dtype=torch.get_default_dtype()))
                terms_2.append(torch.tensor(self.max_dist, device=self.device,
                                            dtype=torch.get_default_dtype()))
                continue

            # Pairwise distances between all possible locations and the GTed locations
            n_gt_pts = gt_b.size()[0]
            normalized_x = norm_factor.repeat(self.n_pixels, 1) *\
                self.all_img_locations
            normalized_y = norm_factor.repeat(len(gt_b), 1)*gt_b
            d_matrix = self.cdist(normalized_x, normalized_y)

            # Reshape probability map as a long column vector,
            # and prepare it for multiplication
            p = prob_map_b.view(prob_map_b.nelement())
            n_est_pts = p.sum()
            p_replicated = p.view(-1, 1).repeat(1, n_gt_pts)

            # Weighted Hausdorff Distance
            term_1 = (1 / (n_est_pts + 1e-6)) * \
                torch.sum(p * torch.min(d_matrix, 1)[0])
            weighted_d_matrix = (1 - p_replicated)*self.max_dist + p_replicated*d_matrix
            minn = self.generalize_mean(weighted_d_matrix,
                                  p=self.p,
                                  dim=0, keepdim=False)
            term_2 = torch.mean(minn)

            # terms_1[b] = term_1
            # terms_2[b] = term_2
            terms_1.append(term_1)
            terms_2.append(term_2)
        terms_1 = torch.stack(terms_1)
        terms_2 = torch.stack(terms_2)

        if self.return_2_terms:
            res = terms_1.mean(), terms_2.mean()
        else:
            res = terms_1.mean() + terms_2.mean()
        loss_smooth = self.smooth_l1(pred_count, true_count)
        final_loss = loss_smooth + res
        return final_loss


    def generalize_mean(self, tensor, dim, p=-9, keepdim=False):
        # """
        # Computes the softmin along some axes.
        # Softmin is the same as -softmax(-x), i.e,
        # softmin(x) = -log(sum_i(exp(-x_i)))

        # The smoothness of the operator is controlled with k:
        # softmin(x) = -log(sum_i(exp(-k*x_i)))/k

        # :param input: Tensor of any dimension.
        # :param dim: (int or tuple of ints) The dimension or dimensions to reduce.
        # :param keepdim: (bool) Whether the output tensor has dim retained or not.
        # :param k: (float>0) How similar softmin is to min (the lower the more smooth).
        # """
        # return -torch.log(torch.sum(torch.exp(-k*input), dim, keepdim))/k
        """
        The generalized mean. It corresponds to the minimum when p = -inf.
        https://en.wikipedia.org/wiki/Generalized_mean
        :param tensor: Tensor of any dimension.
        :param dim: (int or tuple of ints) The dimension or dimensions to reduce.
        :param keepdim: (bool) Whether the output tensor has dim retained or not.
        :param p: (float<0).
        """
        assert p < 0
        res= torch.mean((tensor + 1e-6)**p, dim, keepdim=keepdim)**(1./p)
        return res
    
    def _assert_no_grad(self, variables):
        for var in variables:
            assert not var.requires_grad, \
                "nn criterions don't compute the gradient w.r.t. targets - please " \
                "mark these variables as volatile or not requiring gradients"
        
    def cdist(self, x, y):
        """
        Compute distance between each pair of the two collections of inputs.
        :param x: Nxd Tensor
        :param y: Mxd Tensor
        :res: NxM matrix where dist[i,j] is the norm between x[i,:] and y[j,:],
            i.e. dist[i,j] = ||x[i,:]-y[j,:]||

        """
        differences = x.unsqueeze(1) - y.unsqueeze(0)
        distances = (torch.sum(differences**2, -1) + 1e-06).sqrt()
        return distances
