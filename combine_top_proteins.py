import numpy as np
import torch
import torchvision.transforms.functional as TF
import tifffile
import matplotlib as mpl
mpl.use('Agg')
import matplotlib as plt

import os
import argparse
import pywt
import cv2
import re

from main import SimpleCNN

def format_image(im):
    im = cv2.resize(im, dsize=(1024,1024))    
    
    if np.std(im) != 0:
        im = (im - np.mean(im)) / np.std(im)
        im[im>3] = 3.
        im[im<-3] = -3.
    else:
        im = im - np.mean(im)
    
    coeffs = pywt.dwt2(im, 'db2')
    ll, (lh, hl, hh) = coeffs
    im = np.concatenate((ll[...,None], lh[...,None], hl[...,None], hh[...,None]), axis=2) 
    im = TF.to_tensor(np.float64(im))
    patches = im.unfold(1,128,128).unfold(2,128,128)
    patches = patches.transpose(0,2)
    im = patches.contiguous().view(-1, 4, 128, 128)
    
    return im

parser = argparse.ArgumentParser()
parser.add_argument('--calc', help='True if ranking is already calculated', action='store_true')
parser.add_argument('--create-images', help='set to create the weighted images', action='store_true')
parser.add_argument('--rank', help='location of protein ranking file')
parser.add_argument('--model', '-m', help='model file location')
parser.add_argument('--data', help='data path of focused images')
parser.add_argument('--src', help='data directory to put weighted images')
parser.add_argument('--hastwo', help='file names of src images start with 2', action='store_true')
parser.add_argument('-z',help='number of z-planes in dataset', type=int)
parser.add_argument('--regions',help='number of regions in dataset', type=int)
parser.add_argument('--tiles',help='number of tiles in dataset', type=int)
parser.add_argument('--stanford',help='set if Stanford small/large intestine dataset', action='store_true')
parser.add_argument('--goltsev', help='set if Goltsev et al. dataset', action='store_true')
parser.add_argument('--newcodex', help='set if one of the new Dec2020 UF datasets', action='store_true')
parser.add_argument('--channels',help='file with which channels to consider')
parser.add_argument('--num-weighted', help='number of proteins to use for weighted images', type=int)
args = parser.parse_args()


if args.stanford:
    ch_boolean = np.loadtxt(args.channels, delimiter=',', dtype=str)
    protein_indices = np.array([i for i,item in enumerate(ch_boolean[:,1]) if 'True' in item])
    num_c = 4
    num_t = len(ch_boolean) // num_c # should be 20 for small, 21 for large intestine

    weighted_cyc = 20
    weighted_ch = 2
    
    cyc_capital = 'C'

elif args.goltsev:
    pat_protein = re.compile('(?:protein)(..)') 
    protein_indices = np.arange(0,31)

elif args.newcodex:
    pat_protein = re.compile('(?:protein)(..)')
    ch_boolean = np.loadtxt(args.channels, delimiter=',', dtype=str)
    protein_indices = np.array([i for i,item in enumerate(ch_boolean[:,1]) if 'True' in item])
    num_c = 4
    cyc_capital = 'c'
    weighted_cyc = 12
    weighted_ch = 2

else: # Florida datasets
    protein_indices = np.array([0, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19, 22, 23, 26, 27, 30, 31])
    num_c = 4
    num_t = 9

    weighted_cyc = 9
    weighted_ch = 4

    cyc_capital = 'c'

if args.calc is False: # Need to calculate protein ranking
    device = torch.device('cpu')
    image_shape = (128,128)
    model = SimpleCNN(image_shape)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    image_list = sorted(os.listdir(args.data))
    pat_t = re.compile('(?:t)(...)')
    pat_c = re.compile('(?:c)(...)')

    protein_scores = np.zeros((len(protein_indices),2))

    print('Ranking proteins')
    for i, image_name in enumerate(image_list):
        if args.goltsev:
            protein_idx = int(re.findall(pat_protein, image_name)[0])
            score_idx = protein_idx
        elif args.newcodex:
            protein_idx = int(re.findall(pat_protein, image_name)[0])
            score_idx = list(protein_indices).index(protein_idx)
        else:
            tile_idx = i // len(protein_indices)
            t = int(re.findall(pat_t, image_name)[0])
            c = int(re.findall(pat_c, image_name)[0])
            protein_idx = (t-1)*num_c + (c-1)
            score_idx = list(protein_indices).index(protein_idx)
        
        im = tifffile.imread(os.path.join(args.data, image_name))
        im = format_image(im)
       
        with torch.no_grad():
            output = model(im.view(-1, 4, 128, 128).type('torch.FloatTensor'))

        protein_scores[score_idx, 0] += output.max()
        protein_scores[score_idx, 1] += 1
    
    protein_scores[:,0] /= protein_scores[:,1]

    np.save(args.rank, protein_scores)
else:
    print('Loading previously calculated ranking...')
    protein_scores = np.load(args.rank)
     
sorted_idx = np.argsort(protein_scores[:,0])[::-1]
for j in range(len(protein_indices)):
    print(sorted_idx[j], protein_indices[sorted_idx[j]], protein_scores[:,0][sorted_idx[j]])
print('Top {} protein indices and scores:'.format(args.num_weighted))
for j in range(args.num_weighted):
    print(protein_indices[sorted_idx[j]], protein_scores[:,0][sorted_idx[j]])
            
###-------------------------------####

if args.create_images:

    if args.goltsev:
        ch_boolean = np.loadtxt(args.channels, delimiter=',', dtype=str)
        protein_indices = np.array([i for i,item in enumerate(ch_boolean[:,1]) if 'True' in item])
        assert(len(protein_indices) == 31)
        num_c = 3
        num_t = 18
        weighted_cyc = 18
        weighted_ch = 3
        cyc_capital = 'c'
            
    top_weights = np.zeros((args.num_weighted, 3)) # columns: weight, cycle number, channel number
    print('Creating weighted image of top {} proteins'.format(args.num_weighted))
    for i, idx in enumerate(protein_indices[sorted_idx[:args.num_weighted]]):
        cyc_num = (idx // num_c) + 1
        ch_num = (idx % num_c) + 1
        top_weights[i,:] = np.array([protein_scores[:,0][sorted_idx[i]], cyc_num, ch_num])
    
    for reg_i in range(args.regions):
        print('Region {}'.format(reg_i+1))
        for z in range(args.z):
            print('z = {}'.format(z+1))
            for tile in range(args.tiles):
                if args.hastwo:
                    paths = ['{}yc{:d}_reg{:1d}/2_{:05d}_Z{:03d}_CH{:1d}.tif'.format(cyc_capital, int(top_weights[i,1]), reg_i+1, tile+1, z+1, int(top_weights[i,2])) for i in range(args.num_weighted)]
                elif args.goltsev:
                    paths = ['{}yc{:02d}_reg{:1d}/1_{:05d}_Z{:03d}_CH{:1d}.tif'.format(cyc_capital, int(top_weights[i,1]), reg_i+1, tile+1, z+1, int(top_weights[i,2])) for i in range(args.num_weighted)]
                elif args.newcodex:
                    paths = ['{}yc{:03d}_reg{:03d}/1_{:05d}_Z{:03d}_CH{:1d}.tif'.format(cyc_capital, int(top_weights[i,1]), reg_i+1, tile+1, z+1, int(top_weights[i,2])) for i in range(args.num_weighted)]
                else:
                    paths = ['{}yc{:d}_reg{:1d}/1_{:05d}_Z{:03d}_CH{:1d}.tif'.format(cyc_capital, int(top_weights[i,1]), reg_i+1, tile+1, z+1, int(top_weights[i,2])) for i in range(args.num_weighted)]
                ims = [tifffile.imread(os.path.join(args.src, path)) for path in paths]
                assert(len(ims) == args.num_weighted)   
     
                if args.hastwo:
                    weighted_path = '{}yc{:d}_reg{:d}/2_{:05d}_Z{:03d}_CH{:d}.tif'.format(cyc_capital, weighted_cyc, reg_i+1, tile+1, z+1, weighted_ch)
                elif args.goltsev:
                    weighted_path = '{}yc{:02d}_reg{:d}/1_{:05d}_Z{:03d}_CH{:d}.tif'.format(cyc_capital, weighted_cyc, reg_i+1, tile+1, z+1, weighted_ch)
                elif args.newcodex:
                    weighted_path = '{}yc{:03d}_reg{:03d}/1_{:05d}_Z{:03d}_CH{:d}.tif'.format(cyc_capital, weighted_cyc, reg_i+1, tile+1, z+1, weighted_ch)
                else:
                    weighted_path = '{}yc{:d}_reg{:d}/1_{:05d}_Z{:03d}_CH{:d}.tif'.format(cyc_capital, weighted_cyc, reg_i+1, tile+1, z+1, weighted_ch)
    
                weighted_num = np.sum(np.array( [top_weights[i,0]*ims[i] for i in range(args.num_weighted)] ), 0)
                weighted_norm = np.sum(top_weights[:,0])
        
                weighted_im = weighted_num/weighted_norm
                weighted_im = np.asarray(weighted_im,dtype=ims[0].dtype)
                #print(path1, path2, path3)
                #print(os.path.join(args.src, weighted_path))
        
                with tifffile.TiffWriter(os.path.join(args.src, weighted_path)) as tif:
                    tif.save(weighted_im)
                #print(weighted_path)   
 
    print('Finished creating weighted images to cycle {}, channel {}'.format(weighted_cyc, weighted_ch))


