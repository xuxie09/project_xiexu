import torch
import torch.nn as nn

from utils import r2c, c2r
from models import mri

class ScaleLayer(nn.Module):
    def __init__(self):
        super(ScaleLayer, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([0.1]))  # 正确: nn.Parameter内部使用Tensor

    def forward(self, input):
        return input * self.scale

def Residual_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        ScaleLayer()
    )

class ResNetDenoiser(nn.Module):
    def __init__(self, n_layers):
        super(ResNetDenoiser, self).__init__()
        layers = []
        layers += nn.Sequential(nn.Conv2d(2, 64, 3, padding=1))
        for _ in range(n_layers-2):
            layers += Residual_block(64, 64)
        layers += nn.Sequential(nn.Conv2d(64, 2, 3, padding=1))       
        
        self.nw = nn.Sequential(*layers)

    def forward(self, x):
        idt = x
        dw = self.nw(x) + idt  # Final conv layer and add the input
        return dw
    
class myAtA(nn.Module):
    """
    performs DC step
    """
    def __init__(self, csm, mask, lam):
        super(myAtA, self).__init__()
        self.csm = csm # complex (B x ncoil x nrow x ncol)
        self.mask = mask # complex (B x nrow x ncol)
        self.lam = lam

        self.A = mri.SenseOp(csm, mask)

    def forward(self, im): #step for batch image
        """
        :im: complex image (B x nrow x nrol)
        """
        im_u = self.A.adj(self.A.fwd(im))
        return im_u + self.lam * im

def myCG(AtA, rhs):
    """
    performs CG algorithm
    :AtA: a class object that contains csm, mask and lambda and operates forward model
    """
    rhs = r2c(rhs, axis=1) # nrow, ncol
    x = torch.zeros_like(rhs)
    i, r, p = 0, rhs, rhs
    rTr = torch.sum(r.conj()*r).real
    while i < 10 and rTr > 1e-10:
        Ap = AtA(p)
        alpha = rTr / torch.sum(p.conj()*Ap).real
        alpha = alpha
        x = x + alpha * p
        r = r - alpha * Ap
        rTrNew = torch.sum(r.conj()*r).real
        beta = rTrNew / rTr
        beta = beta
        p = r + beta * p
        i += 1
        rTr = rTrNew
    return c2r(x, axis=1)

class data_consistency(nn.Module):
    def __init__(self):
        super().__init__()
        self.lam = nn.Parameter(torch.tensor(0.05), requires_grad=True)

    def forward(self, z_k, x0, csm, mask):
        rhs = x0 + self.lam * z_k # (2, nrow, ncol)
        AtA = myAtA(csm, mask, self.lam)
        rec = myCG(AtA, rhs)
        return rec

#model =======================
class SSDU(nn.Module):
    def __init__(self, n_layers, k_iters):
        """
        :n_layers: number of layers
        :k_iters: number of iterations
        """
        super().__init__()
        self.k_iters = k_iters
        self.dw = ResNetDenoiser(n_layers)
        self.dc = data_consistency()

    def forward(self, x0, csm, trn_mask):
        """
        :x0: zero-filled reconstruction (B, 2, nrow, ncol) - float32
        :csm: coil sensitivity map (B, ncoil, nrow, ncol) - complex64
        :mask: sampling mask (B, nrow, ncol) - int8
        """

        x_k = x0.clone()
        for k in range(self.k_iters):
            # cnn denoiser
            z_k = self.dw(x_k) # (2, nrow, ncol)
            # data consistency
            x_k = self.dc(z_k, x0, csm, trn_mask) # (2, nrow, ncol)
        # SenseOp = mri.SenseOp(csm, loss_mask)
        # y_k = SenseOp.fwd(x_k)
        return x_k
    