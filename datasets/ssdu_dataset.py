import torch
from torch.utils.data import Dataset
import h5py as h5
import numpy as np

from utils import c2r
from models import mri

class ssdu_dataset(Dataset):
    def __init__(self, mode, dataset_path, sigma=0.01):
        """
        :sigma: std of Gaussian noise to be added in the k-space
        """
        self.prefix = 'trn' if mode == 'train' else 'tst'
        self.dataset_path = dataset_path
        self.sigma = sigma
    def __getitem__(self, index):
        with h5.File(self.dataset_path, 'r') as f:
            gt, csm, mask = f[self.prefix+'Org'][index], f[self.prefix+'Csm'][index], f[self.prefix+'Mask'][index]
        
            if self.prefix == 'trn':
                from ssdu_masks import ssdu_mask
                ssdu = ssdu_mask()
                #np.newaxis 用于将现有数组的维度增加一维，通常用于满足 NumPy 中数组的形状要求。 当应用于 csm 和 mask 时，它可能用于添加批处理维度，这通常是 PyTorch 等深度学习框架中的批处理所需的。
                SenseOp = mri.SenseOp(csm[np.newaxis,:,:,:], mask[np.newaxis,:,:]) # ZT 
                k_space_usamp = SenseOp.fwd(gt[np.newaxis,:,:]) # Fig 2: acquired k-space Omega ZT
                k_space_usamp = k_space_usamp.numpy()

                k_split_mask = np.transpose(k_space_usamp[0], (1, 2, 0))# 例如，如果数据的原始形状是“(N_coil, N_y, N_x)”，其中“N_coil”是线圈数量，“N_y”是 k 空间的 y 维度，“N_x”是 x 维度，转置操作后的新形状将是“(N_y, N_x, N_coil)”。

                trn_mask, loss_mask = ssdu.uniform_selection(k_split_mask, mask)
                trn_mask = trn_mask.astype(np.int8)
                loss_mask = loss_mask.astype(np.int8)
                x0 = undersample_(gt, csm, trn_mask, self.sigma)

                # expanded_loss_mask = np.repeat(loss_mask[:, :, np.newaxis], 16, axis=2)
                # result = expanded_loss_mask * k_space_usamp
                # k_space_lossf = np.transpose(result, (2, 0, 1))
                # k_space_lossf = result[np.newaxis, :]

                k_space_lossf = loss_mask * k_space_usamp # Fig 2: set2 k-space ZT
                return torch.from_numpy(c2r(x0)), torch.from_numpy(c2r(k_space_lossf[0])), torch.from_numpy(c2r(gt)),torch.from_numpy(csm), torch.from_numpy(trn_mask), torch.from_numpy(loss_mask) 
                #`torch.from_numpy` 函数用于将 NumPy 数组转换为 PyTorch 张量。
                #c2r     """:input shape: row x col (complex64):output shape: 2 x row x col (float32)"""
            
            if self.prefix == 'tst':
                with h5.File(self.dataset_path, 'r') as f:
                    gt, csm, mask = f[self.prefix+'Org'][index], f[self.prefix+'Csm'][index], f[self.prefix+'Mask'][index]

                x0 = undersample_(gt, csm, mask, self.sigma)
                return torch.from_numpy(c2r(x0)), torch.from_numpy(c2r(gt)), torch.from_numpy(csm), torch.from_numpy(mask)
            
            
    def __len__(self):
        with h5.File(self.dataset_path, 'r') as f:
            num_data = len(f[self.prefix+'Mask'])
        return num_data


def undersample_(gt, csm, mask, sigma):

    ncoil, nrow, ncol = csm.shape
    csm = csm[None, ...]  # 4dim

    # shift sampling mask to k-space center
    mask = np.fft.ifftshift(mask, axes=(-2, -1))

    SenseOp = mri.SenseOp(csm, mask)

    b = SenseOp.fwd(gt)

    noise = torch.randn(b.shape) + 1j * torch.randn(b.shape)
    noise = noise * sigma / (2.**0.5)

    atb = SenseOp.adj(b + noise).squeeze(0).detach().numpy()

    return atb