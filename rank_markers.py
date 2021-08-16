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
parser.add_argument('--num-cycles', type=int, help = 'the number of cycles present in the data', required = True)
parser.add_argument('--num-channels-per-cycle', type=int, help = 'the number of channels per cycle. For example, if there are 10 cycles with 3 channels each, then we would expect a total of 30 channels.', required = True)
parser.add_argument('--no-ranking', help = 'set this flag if ranking has already been calculated. If this is set, then --model-pathargument does not need to be specified', action = 'store_true')
parser.add_argument('-r', '--rank-path', help = "path to file where marker ranking is to be saved/where marker ranking is saved (if ranking has already been performed). IMPORTANT: the file extension should be '.csv'", required = True)
parser.add_argument('--gpu', action='store_true', help = 'set if you want to use the GPU')

# Weighted image arguments
parser.add_argument('--create-images', help = 'set this flag to create weighted images based on the top --num_weighted markers. If this flag is set, the --num-weighted argument must be specified.', action = 'store_true')
parser.add_argument('--num-weighted', type=int, help = 'number of top-ranked markers to use to create weighted images', default = 3)
parser.add_argument('--output-weighted', help = 'path to directory to output weighted images. Must be specified if the --create-images flag is set.')
parser.add_argument('--exclude', nargs='*', type=int, help = 'rank of any markers that you wish to exclude from the combined weighted images.')

args = parser.parse_args()

if (args.create_images  and (not args.num_weighted or not args.output_weighted)):
    parser.error('The --create-images argument requires the --num-weighted and the --output-weighted arguments to be specified.')

# We expect each row of the channel csv file to have Name, Boolean
ch_boolean = np.loadtxt(args.channels, delimiter=',', dtype=str)
marker_indices = np.array([i for i,item in enumerate(ch_boolean[:,1]) if 'True' in item]) # which indices out of all the channels to use
num_markers = len(marker_indices)

if args.no_ranking is False: # Ranking has not been calculated

    if (not args.model_path or not args.data_dir):
        parser.error('Unless the --no-ranking flag is set, we require the --model-path and --data-dir arguments to be specified.')

    if args.gpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    image_shape = (128,128)
    model = SimpleCNN(image_shape)
    model.load_state_dict(torch.load(args.model_path, map_location = device))
    model.eval()
    
    marker_scores = np.zeros((num_markers, 2)) # the score for each marker of interest
    
    image_list = sorted(os.listdir(args.data_dir))
    pat_t = re.compile('(?:t)(...)')
    pat_c = re.compile('(?:c)(...)')
    
    # Ranking proteins
    for i, image_file in enumerate(image_list):
        
        t = int(re.findall(pat_t, image_file)[0]) # cycle number, starts from 1
        c = int(re.findall(pat_c, image_file)[0]) # channel number, starts from 1
        marker_idx = (t-1)*args.num_channels_per_cycle + (c-1) # which marker index this image refers to
        if marker_idx not in marker_indices:
            continue
        score_idx = list(marker_indices).index(marker_idx) # which index we need to use for the marker_scores array
        
        im = tifffile.imread(os.path.join(args.data_dir, image_file))
        im = preprocessImage(im)
        
        with torch.no_grad():
            output = model(im.view(-1, 4, 128, 128).type('torch.FloatTensor'))

        marker_scores[score_idx, 0] += output.max()
        marker_scores[score_idx, 1] += 1

    marker_scores[:,0] /= marker_scores[:,1] # averaging over all tiles
    
    # Output rank and scores
    sorted_idx = np.argsort(marker_scores[:,0])[::-1] # sorted indices of the marker_scores, len = num_markers
    score_dict = {'Marker': ch_boolean[marker_indices[sorted_idx], 0], 'Score': marker_scores[sorted_idx,0], 'Cycle': marker_indices[sorted_idx]//args.num_channels_per_cycle + 1, 'Channel': marker_indices[sorted_idx]%args.num_channels_per_cycle + 1}
    score_df = pd.DataFrame(data=score_dict)
    score_df.to_csv(args.rank_path, index=False)

else:
    # Ranking is already calculated
    score_df = pd.read_csv(args.rank_path)
    reverse_indices = np.array([list(ch_boolean[:,0]).index(i) for i in score_df['Marker'].values])
    sorted_idx = np.argsort(np.argsort(reverse_indices))

print(score_df)


if args.create_images:
    print()
    print('Creating weighted images based on the top {} markers'.format(args.num_weighted))

    top_weights = np.zeros((args.num_weighted, 3))
    if args.exclude is not None:
        sorted_idx = np.delete(sorted_idx, np.array(args.exclude)-1)
    for i, idx in enumerate(marker_indices[sorted_idx[:args.num_weighted]]):
        cyc_num = (idx // args.num_channels_per_cycle) + 1 
        ch_num = (idx % args.num_channels_per_cycle) + 1 
        top_weights[i,:] = np.array([score_df.iloc[i,1], cyc_num, ch_num])

    image_list = sorted(os.listdir(args.data_dir))
    pat_t = re.compile('(?:t)(...)')
    pat_c = re.compile('(?:c)(...)')


    req_pat = re.compile('(t\d{3}.c\d{3}|c\d{3}.t\d{3})', flags = re.IGNORECASE)
    tc_pat = re.findall(req_pat, image_list[0])[0]
    _, tif_ext = os.path.splitext(image_list[0])
    tile_ids = []

    for i, image_file in enumerate(image_list):
        tile_ids.append(''.join(re.sub(req_pat, '', image_file).split('.')[:-1]))
    
    if not os.path.exists(args.output_weighted):
        os.makedirs(args.output_weighted)

    tile_ids = list(set(tile_ids))
    for tile_id in tile_ids:
        paths = []
        for marker_i in range(args.num_weighted):
            t = int(top_weights[marker_i, 1])
            c = int(top_weights[marker_i, 2])
            tc_pat = re.sub(r'(?:t)(\d{3})', f't{t:03d}', tc_pat)
            tc_pat = re.sub(r'(?:c)(\d{3})', f'c{c:03d}', tc_pat)
            
            paths.append(f'{tile_id}{tc_pat}{tif_ext}')
        ims = [tifffile.imread(os.path.join(args.data_dir, path)) for path in paths]
        
        weighted_num = np.sum(np.array( [top_weights[i,0]*ims[i] for i in range(args.num_weighted)] ), 0)
        weighted_norm = np.sum(top_weights[:,0])
    
        weighted_im = weighted_num/weighted_norm
        weighted_im = np.asarray(weighted_im,dtype=ims[0].dtype)

        weighted_path = os.path.join(args.output_weighted,f'{tile_id}weighted{tif_ext}') 
        
        with tifffile.TiffWriter(weighted_path) as tif:
            tif.save(weighted_im)

    print()
    print(f'Weighted images saved to {args.output_weighted}')



