import torch
import torchvision.transforms.functional as TF
import numpy as np
import pandas as pd
import pywt
import tifffile

import glob
import os
import cv2
import re
import argparse

from ramces_cnn import SimpleCNN

def preprocessImage(im):
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

# Ranking arguments
parser.add_argument('-m', '--model-path', help = 'path to trained model to rank markers. Does not need to be specified if --no-ranking is set.', default = './models/trained_model.h5')
parser.add_argument('-d', '--data-dir', help = 'path to directory with image data. Each image file should represent a SINGLE channel, with the file name in the required format as detailed on github.com/mdayao/ramces.')
parser.add_argument('--channels', help = 'path to file indicating which channels to use and their names. See github.com/mdayao/ramces for details on how to format this file.', required = True)
parser.add_argument('--num-cycles', help = 'the number of cycles present in the data', required = True)
parser.add_argument('--num-channels-per-cycle', help = 'the number of channels per cycle. For example, if there are 10 cycles with 3 channels each, then we would expect a total of 30 channels.', required = True)
parser.add_argument('--no-ranking', help = 'set this flag if ranking has already been calculated. If this is set, then --model-pathargument does not need to be specified', action = 'store_true')
parser.add_argument('-r', '--rank-path', help = "path to file where marker ranking is to be saved/where marker ranking is saved (if ranking has already been performed). IMPORTANT: the file extension should be '.csv'", required = True)

# Weighted image arguments
parser.add_argument('--create-images', help = 'set this flag to create weighted images based on the top --num_weighted markers. If this flag is set, the --num-weighted argument must be specified.', action = 'store_true')
parser.add_argument('--num-weighted', help = 'number of top-ranked markers to use to create weighted images', default = 3)
parser.add_argument('--output-weighted', help = 'path to directory to output weighted images. Must be specified if the --create-images flag is set.')

args = parser.parse_args()

if (args.create_images  and (not args.num_weighted or not args.output_weighted)):
    parser.error('The --create-images argument requires the --num-weighted and the --output-weighted arguments to be specified.')

if args.no_ranking is False: # Ranking has not been calculated

    if (not args.model_path or not args.data_dir):
        parser.error('Unless the --no-ranking flag is set, we require the --model-path and --data-dir arguments to be specified.')

    device = torch.device('cpu')
    
    image_shape = (128,128)
    model = SimpleCNN(image_shape)
    model.load_state_dict(torch.load(args.model_path, map_location = device))
    model.eval()
    
    # We expect each row of the channel csv file to have Name, Boolean
    ch_boolean = np.loadtxt(args.channels, delimiter=',', dtype=str)
    marker_indices = np.array([i for i,item in enumerate(ch_boolean[:,1]) if 'True' in item]) # which indices out of all the channels to use
    num_markers = len(marker_indices)
    marker_scores = np.zeros((num_markers, 2)) # the score for each marker of interest
    
    image_list = sorted(os.listdir(args.data_dir))
    pat_t = re.compile('(?:t)(...)')
    pat_c = re.compile('(?:c)(...)')
    
    # Ranking proteins
    for i, image_file in enumerate(image_list):
        
        t = int(re.findall(pat_t, image_file)[0]) # cycle number, starts from 1
        c = int(re.findall(pat_c, image_file)[0]) # channel number, starts from 1
        marker_idx = (t-1)*args.num_channels_per_cycle + (c-1) # which marker index this image refers to
        score_idx = list(marker_indices).index(marker_idx) # which index we need to use for the marker_scores array
        
        im = tiffile.imread(os.path.join(args.data_dir, image_file))
        im = preprocessImage(im)
        
        with torch.no_grad():
            output = model(im.view(-1, 4, 128, 128).type('torch.FloatTensor'))

        marker_scores[score_idx, 0] += output.max()
        marker_scores[score_idx, 1] += 1

    marker_scores[:,0] /= marker_scores[:,1] # averaging over all tiles
    
    # Output rank and scores
    sorted_idx = np.argsort(marker_scores[:,0])[::-1] # sorted indices of the marker_scores, len = num_markers
    score_dict = {'Marker': ch_boolean[marker_indices[sorted_idx], 0], 'Score': marker_scores[sorted_idx,0]}
    score_df = pd.DataFrame(data=score_dict)
    score_df.to_csv(args.rank_path, index=False)

else:
    # Ranking is already calculated
    score_df = pd.read_csv(args.rank_path)

print(score_df)

#if args.create_images:

    # TODO: write code to create weighted images based on ranking


