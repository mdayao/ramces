import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import numpy as np
from skimage.external import tifffile
import matplotlib.pyplot as plt
import pywt
import glob
import os
import cv2
from main import SimpleCNN

device = torch.device('cpu')
data_dir = './data/raw_extracted'
#image_shape = (385, 385)
#image_shape = (505, 673) 
image_shape = (128,128) 
  
model = SimpleCNN(image_shape)
model.load_state_dict(torch.load('./saved_models/dark-sky-95_wavelet_500_patches_dropout.h5', map_location=device))
#model.load_state_dict(torch.load('./saved_models/rose-elevator_wavelet_1000.h5', map_location=device))
#model.load_state_dict(torch.load('./saved_models/twilight-lion-78_wavelet_500.h5', map_location=device))
model.eval()

os.chdir(data_dir)
#image_shape = tifffile.imread('X00_Y00_protein00.tif').shape
#output_image = np.zeros(image_shape)
num_proteins = 31
protein_scores = np.zeros(num_proteins)
for i in range(num_proteins):
    image_template = '*protein{:02d}.tif'.format(i)
    num_im = 0
    for idx, image_file in enumerate(sorted(glob.glob(image_template))):
        if idx % 3 == 2:
            orig_im = tifffile.imread(image_file)
            orig_im = cv2.resize(orig_im, dsize=(1024,1024))     

            #orig_im = (orig_im - np.min(orig_im)) * 1./(np.max(orig_im) - np.min(orig_im))
            im = (orig_im - np.mean(orig_im)) / np.std(orig_im)
            im[im>3] = 3.
            im[im<-3] = -3.
            #im = orig_im[0:256*3, 0:256*3]

            coeffs = pywt.dwt2(im, 'db2')
            ll, (lh, hl, hh) = coeffs
            input_im = np.concatenate((ll[...,None], lh[...,None], hl[...,None], hh[...,None]), axis=2) 
            input_im = TF.to_tensor(np.float64(input_im))

            patches = input_im.unfold(1,128,128).unfold(2,128,128)
            patches = patches.transpose(0,2)
            input_im = patches.contiguous().view(-1, 4, 128, 128)

            with torch.no_grad():
                output = model(input_im.view(-1, 4, 128, 128).type('torch.FloatTensor'))
                #output = model(input_im.view(-1, 4, 505, 673).type('torch.FloatTensor'))

            protein_scores[i] += output.mean()
            num_im += 1
    protein_scores[i] /= num_im

sorted_idx = np.argsort(protein_scores)[::-1]
for j in range(31):
    print(sorted_idx[j], protein_scores[sorted_idx[j]])


