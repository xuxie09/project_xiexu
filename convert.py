import h5py
import sigpy as sp
import matplotlib.pyplot as plt
from sigpy.mri import app
import torch
import numpy as np
x=torch.cuda.is_available()
print(x)

import os
import random
from sigpy.mri import samp
from ssdu_masks import ssdu_mask

def Choose_data(path, datatype):
    if datatype == 't1':
        files = [f for f in os.listdir(path) if f.startswith('file_brain_AXT1_')]
    elif datatype == 't2':
        files = [f for f in os.listdir(path) if f.startswith('file_brain_AXT2_')]
    elif datatype == 'FLAIR':
        files = [f for f in os.listdir(path) if f.startswith('file_brain_AXFLAIR_')]
    elif datatype == 'PRE':
        files = [f for f in os.listdir(path) if f.startswith('file_brain_AXT1PRE_')]
    # 随机挑选30个文件
    selected_files = random.sample(files, min(100, len(files)))
    valid_files = []
    for filename in selected_files:
        file = os.path.join(path, filename)
        with h5py.File(file, 'r') as f:
            k = f['kspace']
            if k.shape == (16, 16, 768, 396):
                valid_files.append(filename)
    
    # 随机挑选15个文件
    new_selected_files = random.sample(valid_files, min(15, len(valid_files)))
    new_selected_files = [path + file for file in new_selected_files]
   

    train_data = new_selected_files[:12]
    test_data = new_selected_files[12:]
    return train_data, test_data


def Org(file):
    with h5py.File(file, 'r') as f:
        k = f['kspace'][:]
    recon = sp.rss(sp.ifft(k, axes=[-2, -1]), axes=(-3))
    combine_recon = sp.resize(recon, oshape = [k.shape[0], k.shape[2], k.shape[3]]) 
    combine_recon = combine_recon / np.linalg.norm(combine_recon) * 100
    return combine_recon

def Mask(type,file):
    s,y,x = Org(file).shape
    if type == 'catesian':   
        mask_cartes = np.zeros([s,y,x])
        mask_cartes[::6, :] = 1
        us_mask = np.fft.fftshift(mask_cartes, axes=(-2, -1))
    elif type == 'possion':
        mask = samp.poisson([y, x], 6)
        us_mask = np.tile(mask, (s, 1, 1))
        us_mask = np.fft.fftshift(us_mask, axes=(-2, -1))
    # elif type == 'ssdu':
    #     us_mask = ssdu_masks.Gaussian_selection().trn_mask
    return us_mask

def Csm(file):
    with h5py.File(file, 'r') as f:
        k = f['kspace'][:]
        device = sp.Device(0) if torch.cuda.is_available() else sp.cpu_device
        print('> device: ', device)
        kspace_dev = sp.to_device(k, device=device)
        cs = []
        for s in range(k.shape[0]):
            k = kspace_dev[s]
            c =app.EspiritCalib(k, device=device).run()
            cs.append(sp.to_device(c))
    return np.array(cs)
    
def main(files,mask_type):
    
    all_Org = []
    all_Mask =[]
    all_Csm = []
    for i in files:
        org = Org(i).astype(np.complex64)
        mask = Mask(mask_type,i).astype(np.int8)
        csm = Csm(i).astype(np.complex64)

        all_Org.append(org)
        all_Csm.append(csm)
        all_Mask.append(mask)
    o = np.concatenate(all_Org, axis=0)
    c = np.concatenate(all_Csm, axis=0)
    m = np.concatenate(all_Mask, axis=0)
    
    return o, c, m





file = '/home/janus/iwbi-cip-datasets/shared/fastMRI/brain/multicoil_train/'
train,test = Choose_data(file, 't2')
train_Org, train_Csm, train_Mask = main(train,'catesian')
test_Org, test_Csm, test_Mask = main(test,'catesian')

with h5py.File('fastmri_data_catesian.h5', 'w') as f:
    f.create_dataset('trnOrg', data=train_Org)
    f.create_dataset('trnCsm', data=train_Csm)
    f.create_dataset('trnMask', data=train_Mask)
    f.create_dataset('tstOrg', data=test_Org)
    f.create_dataset('tstCsm', data=test_Csm)
    f.create_dataset('tstMask', data=test_Mask)
