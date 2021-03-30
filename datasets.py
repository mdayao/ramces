import os
import re

import pandas as pd
import numpy as np
import torch

from torch.utils.data import Dataset
from torchvision import transforms, utils
import torchvision.transforms.functional as TF
from skimage.external import tifffile

import matplotlib.pyplot as plt
import random
import pywt
import cv2

import argparse

class MembraneImageDataset(Dataset):
    
    def __init__(self, label_csv, z_csv, data_dir, trainset=True):
        self.label_array = pd.read_csv(label_csv).to_numpy() # frame, channel, label
#        self.z_frame = pd.read_csv(z_csv) # tile_x, tile_y, best_z
        self.data_dir = data_dir # location of the raw data
        self.image_list = [] # list of strings of raw image file names
        self.protein_pat = re.compile('(?:protein)(..)')  
        self.trainset=trainset
        #self.image_pattern = re.compile('(?:X)(..)(?:_Y)(..)')
        
        #image_tile_idxs = re.findall(self.image_pattern, ' '.join(sorted(os.listdir(data_dir))))
        #image_tile_idxs = np.asarray([[int(i[0]), int(i[1])] for i in image_tile_idxs])

        for i, image_name in enumerate(sorted(os.listdir(data_dir))):
            if trainset:
                if (i // 31) % 3 < 2:
                    self.image_list.append(image_name)
            else:
                if (i // 31) % 3 == 2:
                    self.image_list.append(image_name) 
        
        # adding black images
        if self.trainset:
            self.image_list.extend(['black']*50)

#        self.tile_pattern = re.compile('(?:BALBc-1_X)(..)(?:_Y)(..)') # pattern to extract tile number

#        for i, tile in enumerate(sorted(os.listdir(data_dir))):
#            if trainset:
#                if (i % 3) < 2:
#                    self.tile_list.append(tile)
#            else:
#                if (i % 3) == 2:
#                    self.tile_list.append(tile)

    def __len__(self):
        return len(self.image_list)

    def random_crop(self, img, new_height, new_width):
        height, width = img.shape
        start_h = random.randint(0, height - new_height - 1)
        start_w = random.randint(0, width - new_width - 1)
        return img[start_h:start_h+new_height, start_w:start_w+new_width]
    
    def crop_center(self, img, new_height, new_width):
        height, width = img.shape
        start_h = height // 2 - (new_height // 2)
        start_w = width // 2 - (new_width // 2)
        return img[start_h:start_h+new_height, start_w:start_w+new_width]

    def __getitem__(self, idx):
        #idx = idx // 16
        #patch = idx % 16
        blacked = False
        if self.image_list[idx] == 'black':
            im = np.zeros((1024,1024))
            label = 0
            protein_idx = -1
            blacked = True
        else:
            protein_idx = int(re.findall(self.protein_pat, self.image_list[idx])[0])
            label = self.label_array[protein_idx, 2]
            im = tifffile.imread(os.path.join(self.data_dir, self.image_list[idx]))
            
            if random.random() > 0.5:
                im = self.random_crop(im, 256*3, 256*3) 
            # resize to 1024x1024
            im = cv2.resize(im, dsize=(1024,1024))     
        
            #x_patch = patch % 4
            #y_patch = patch // 4
            #im = im[256*y_patch:256*y_patch+256, 256*x_patch: 256*x_patch+256] 

            #if self.trainset:
            ## randomly set patches to zero to help with sparsity
            ##elif random.random() > 0.5:
            #    rep = 1024 // 4
            #    zero_patches = np.random.choice([0,1], 16, [0.1, 0.9]).reshape((4,4))
            #    zero_patches = zero_patches.repeat(rep, axis=1).repeat(rep, axis=0)
            #    im = im*zero_patches
 
        # normalizing via zscore. Removing outliers using values outside (-3, 3)
        if not blacked and np.std(im) != 0:
            im = (im - np.mean(im)) / np.std(im)
            im[im>3] = 3.
            im[im<-3] = -3.
        else:
            im = im - np.mean(im)

        # normalizing by restricting values to be within 0 and 1
        #im = (im - np.min(im)) * 1./(np.max(im) - np.min(im))
        
        #im = self.crop_center(im, 256*3, 256*3)
 
        if self.trainset and (blacked==False):
            #if random.random() > 0.5:
            #    im = self.random_crop(im, 256*3, 256*3)
            #    im = cv2.resize(im, dsize=(1024,1024))
            if random.random() > 0.5:
                im = np.flipud(im)
            if random.random() > 0.5:
                im = np.fliplr(im)
            im = np.rot90(im, k=random.randint(0,3))
            im = im.copy()

        coeffs = pywt.dwt2(im, 'db2')
        ll, (lh, hl, hh) = coeffs
        im = np.concatenate((ll[...,None], lh[...,None], hl[...,None], hh[...,None]), axis=2) 
        im = TF.to_tensor(np.float64(im))
        patches = im.unfold(1,128,128).unfold(2,128,128)
        patches = patches.transpose(0,2)
        im = patches.contiguous().view(-1, 4, 128, 128)

        return im, label, protein_idx

class CrossValMembraneImageDataset(Dataset):
    
    def __init__(self, label_csvs, data_dirs, florida_sets, trainset=True, all_florida=False):
        self.image_list = [] # list of strings of raw image file names
        self.lens = []
        self.all_florida = all_florida        
        self.trainset=trainset
 
        self.florida_proteins = [0, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19, 22, 23, 26, 27, 30, 31]
        self.florida_setnames = ['lymphnode', 'spleen', 'thymus']
        self.florida_sets = florida_sets # list of strings of which florida set we are using (spleen, thymus, lymphnode)        
        self.florida_pat_t = re.compile('(?:t)(...)')
        self.florida_pat_c = re.compile('(?:c)(...)')

        if trainset:
            if not all_florida:
                self.label_array = pd.read_csv(label_csvs[0]).to_numpy() # frame, channel, label
                self.data_dir = data_dirs[0] # location of the raw data, Goltsev spleen dataset
                self.florida_data_dirs = data_dirs[1:]
                self.protein_pat = re.compile('(?:protein)(..)')  
                self.florida_label_array = pd.read_csv(label_csvs[1], sep = ' ', header=None).to_numpy() # florida dataset: lymphnode, spleen, thymus. rows are proteins
                
                for i, image_name in enumerate(sorted(os.listdir(self.data_dir))):
                    self.image_list.append(('goltsev', image_name))
                self.lens.append(len(os.listdir(self.data_dir)))        

                for ix, florida_dir in enumerate(self.florida_data_dirs):
                    self.lens.append(len(os.listdir(florida_dir)))
                    for i, image_name in enumerate(sorted(os.listdir(florida_dir))):
                        self.image_list.append((florida_sets[ix], image_name))
            
            else: 
                self.florida_label_array = pd.read_csv(label_csvs, sep=' ', header=None).to_numpy() # florida dataset: lymphnode, spleen, thymus. rows are proteins
                self.florida_data_dirs = data_dirs
                
                for ix, florida_dir in enumerate(self.florida_data_dirs):
                    self.lens.append(len(os.listdir(florida_dir)))
                    for i, image_name in enumerate(sorted(os.listdir(florida_dir))):
                        self.image_list.append((florida_sets[ix], image_name))
        
            self.image_list.extend([('black', 'black')]*100)

        else:
            if all_florida:
                self.label_array = pd.read_csv(label_csvs[0]).to_numpy() # frame, channel, label
                self.data_dir = data_dirs[0] # location of the raw data, Goltsev spleen dataset
                self.protein_pat = re.compile('(?:protein)(..)')  
                for i, image_name in enumerate(sorted(os.listdir(self.data_dir))):
                    self.image_list.append(('goltsev', image_name))
            else:
                self.florida_label_array = pd.read_csv(label_csvs, sep=' ', header=None).to_numpy() # florida dataset: lymphnode, spleen, thymus. rows are proteins
                self.florida_data_dirs = data_dirs[0]
                
                for i, image_name in enumerate(sorted(os.listdir(self.florida_data_dirs))):
                    self.image_list.append((florida_sets[0], image_name))
            

    def __len__(self):
        return len(self.image_list)
    
    def random_crop(self, img, new_height, new_width):
        height, width = img.shape
        start_h = random.randint(0, height - new_height - 1)
        start_w = random.randint(0, width - new_width - 1)
        return img[start_h:start_h+new_height, start_w:start_w+new_width]

    def __getitem__(self, idx):
        blacked = False
        protein_idx = -1
        num_c = 4

        if self.trainset:
            if self.image_list[idx][0] == 'black':
                im = np.zeros((1024,1024))
                label = 0
                protein_idx = -1
                blacked = True
            else:
                if not self.all_florida:
                    if self.image_list[idx][0] == self.florida_sets[0]: # 1st florida dataset
                        t = int(re.findall(self.florida_pat_t, self.image_list[idx][1])[0])
                        c = int(re.findall(self.florida_pat_c, self.image_list[idx][1])[0])
                        fprotein_idx = (t-1)*num_c + (c-1)
                        lab_idx = self.florida_proteins.index(fprotein_idx)
                        dataset_idx = self.florida_setnames.index(self.florida_sets[1])
                        label = self.florida_label_array[lab_idx, dataset_idx]
                        im = tifffile.imread(os.path.join(self.florida_data_dirs[0], self.image_list[idx][1]))
                    elif self.image_list[idx][0] == self.florida_sets[1]: # 2nd florida dataset
                        t = int(re.findall(self.florida_pat_t, self.image_list[idx][1])[0])
                        c = int(re.findall(self.florida_pat_c, self.image_list[idx][1])[0])
                        fprotein_idx = (t-1)*num_c + (c-1)
                        lab_idx = self.florida_proteins.index(fprotein_idx)
                        dataset_idx = self.florida_setnames.index(self.florida_sets[0])
                        label = self.florida_label_array[lab_idx, dataset_idx]
                        im = tifffile.imread(os.path.join(self.florida_data_dirs[1], self.image_list[idx][1]))
                    else: # original dataset
                        protein_idx = int(re.findall(self.protein_pat, self.image_list[idx][1])[0])
                        label = self.label_array[protein_idx, 2]
                        im = tifffile.imread(os.path.join(self.data_dir, self.image_list[idx][1]))
                else:
                    t = int(re.findall(self.florida_pat_t, self.image_list[idx][1])[0])
                    c = int(re.findall(self.florida_pat_c, self.image_list[idx][1])[0])
                    fprotein_idx = (t-1)*num_c + (c-1)
                    lab_idx = self.florida_proteins.index(fprotein_idx)
                    if self.image_list[idx][0] == self.florida_sets[0]: # 1st florida dataset
                        dataset_idx = self.florida_setnames.index(self.florida_sets[0])
                        label = self.florida_label_array[lab_idx, dataset_idx]
                        im = tifffile.imread(os.path.join(self.florida_data_dirs[0], self.image_list[idx][1]))
                    elif self.image_list[idx][0] == self.florida_sets[1]: # 2nd florida dataset
                        dataset_idx = self.florida_setnames.index(self.florida_sets[1])
                        label = self.florida_label_array[lab_idx, dataset_idx]
                        im = tifffile.imread(os.path.join(self.florida_data_dirs[1], self.image_list[idx][1]))
                    else: # 3rd florida dataset
                        dataset_idx = self.florida_setnames.index(self.florida_sets[2])
                        label = self.florida_label_array[lab_idx, dataset_idx]
                        im = tifffile.imread(os.path.join(self.florida_data_dirs[2], self.image_list[idx][1]))
                
                if random.random() > 0.5:
                    im = self.random_crop(im, 256*3, 256*3) 
        else:
            if self.all_florida:
                protein_idx = int(re.findall(self.protein_pat, self.image_list[idx][1])[0])
                label = self.label_array[protein_idx, 2]
                im = tifffile.imread(os.path.join(self.data_dir, self.image_list[idx][1]))
            else:
                t = int(re.findall(self.florida_pat_t, self.image_list[idx][1])[0])
                c = int(re.findall(self.florida_pat_c, self.image_list[idx][1])[0])
                fprotein_idx = (t-1)*num_c + (c-1)
                lab_idx = self.florida_proteins.index(fprotein_idx)
                dataset_idx = self.florida_setnames.index(self.florida_sets[0])
                label = self.florida_label_array[lab_idx, dataset_idx]
                im = tifffile.imread(os.path.join(self.florida_data_dirs, self.image_list[idx][1]))

        # resize to 1024x1024
        im = cv2.resize(im, dsize=(1024,1024))     
 
        # normalizing via zscore. Removing outliers using values outside (-3, 3)
        if not blacked and np.std(im) != 0:
            im = (im - np.mean(im)) / np.std(im)
            im[im>3] = 3.
            im[im<-3] = -3.
        else:
            im = im - np.mean(im)
 
        if self.trainset and (blacked==False):
            if random.random() > 0.5:
                im = np.flipud(im)
            if random.random() > 0.5:
                im = np.fliplr(im)
            im = np.rot90(im, k=random.randint(0,3))
            im = im.copy()

        coeffs = pywt.dwt2(im, 'db2')
        ll, (lh, hl, hh) = coeffs
        im = np.concatenate((ll[...,None], lh[...,None], hl[...,None], hh[...,None]), axis=2) 
        im = TF.to_tensor(np.float64(im))
        patches = im.unfold(1,128,128).unfold(2,128,128)
        patches = patches.transpose(0,2)
        im = patches.contiguous().view(-1, 4, 128, 128)
        #im = TF.to_tensor(np.float64(ll))
        #patches = im.unfold(1,128,128).unfold(2, 128, 128)
        #im = patches.contiguous().view(-1, 128, 128)

        return im, label

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', default='orig', help='orig or crossval')
    args = parser.parse_args()

    if args.dataset == 'orig':
        dataset = MembraneImageDataset('./data/class_labels.csv', './data/best_z.csv', './data/raw_extracted')
        print(len(dataset))
        testset = MembraneImageDataset('./data/class_labels.csv', './data/best_z.csv', './data/raw_extracted', trainset=False)
        print(len(testset))
        print(np.asarray(dataset[399][0]).shape)
        print(dataset[399][2])
        print(np.asarray(dataset[399][0]).shape)
        print(dataset[339][1])
        #plt.imshow(np.asarray(testset[399][0])[5,0,:,:], cmap='gray')
        #plt.show()
        #plt.clf()
    elif args.dataset == 'crossval':
        dataset = CrossValMembraneImageDataset('./florida/labels.csv', ['./florida/spleen_filtered', './florida/lymphnode_filtered','./florida/thymus_filtered'], ['spleen', 'lymphnode', 'thymus'], trainset=True, all_florida=True)
        print(len(dataset))
        image_view = 1937
        print(dataset[image_view][1], dataset.image_list[image_view])
        #plt.imshow(np.asarray(dataset[image_view][0])[5,0,:,:], cmap='gray')
        #plt.show()
        #plt.clf()
        test = CrossValMembraneImageDataset('./florida/labels.csv', ['./florida/thymus_filtered'], ['thymus'], trainset=False, all_florida=False)
        print(len(test))
        print(test[100][1], test.image_list[100])
        print(test[100][0].shape)
        #plt.imshow(np.asarray(test[100][0])[5,0,:,:],cmap='gray')
        #plt.show()
    else:
        print('enter orig or crossval for --dataset (-d) tag')


