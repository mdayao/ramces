import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np

from datasets import MembraneImageDataset, CrossValMembraneImageDataset

import wandb
import argparse

class SimpleCNN(nn.Module):

    def __init__(self, image_dim):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
            )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
            )
        
        new_width = image_dim[1] // 8 
        new_height = image_dim[0] // 8
        self.last_layer_size = new_width * new_height * 128
        self.dropout = nn.Dropout(p=.3)
        self.fc1 = nn.Linear(self.last_layer_size, 1)
        #self.fc2 = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()
        

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out.view(-1, self.last_layer_size)
        out = self.dropout(out)
        out = self.fc1(out)
        #out = self.fc2(out)
        out = self.sigmoid(out)
        return out

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data)        

def confusion_matrix(preds, labels, tp, tn, fp, fn):
    conf_matrix = np.zeros((2,2))
    for pred, label in zip((preds, labels)):
        conf_matrix[int(pred), int(label)] += 1
    tp += conf_matrix[1,1]
    tn += conf_matrix[0,0]
    fp += conf_matrix[1,0]
    fn += conf_matrix[0,1]
    return tp, tn, fp, fn

def train_model(model, device, batch_size, trainloader, testloader, num_epochs, criterion, optimizer):

    for epoch in range(num_epochs):
        
        train_out_dist = []
           
        model.train()
        train_loss = 0.0
        train_correct = 0.0
        for batch_idx, data in enumerate(trainloader):
            inputs, targets = data[0].to(device).float(), data[1].to(device).float()
            targets = targets[...,None].expand(-1,16).flatten(-2,-1)
            targets.unsqueeze_(1)
            optimizer.zero_grad()
            inputs = inputs.view(-1, 4, 128, 128)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            train_out_dist += outputs.flatten().tolist()
            pred = outputs >= 0.5
            pred = pred.type(torch.FloatTensor).to(device)
            train_correct += (pred == targets.view_as(pred)).sum().item()

        train_acc = train_correct / (len(trainloader.dataset)*16)
        train_loss = train_loss / (len(trainloader.dataset))        
        
        test_out_dist = []
        test_loss = 0.0
        test_correct = 0.0
        model.eval()
        with torch.no_grad():
            for data in testloader:
                inputs, targets = data[0].to(device).float(), data[1].to(device).float()
                targets = targets[...,None].expand(-1,16).flatten(-2,-1)
                targets.unsqueeze_(1)
                #inputs = inputs.view(-1, 1, 128, 128)
                inputs = inputs.view(-1, 4, 128, 128)
                outputs = model(inputs)
                test_out_dist += outputs.flatten().tolist()
                test_loss += criterion(outputs, targets)
                pred = outputs >= .5
                pred = pred.type(torch.FloatTensor).to(device)
                test_correct += (pred == targets.view_as(pred)).sum().item()
        
        test_loss = test_loss / (len(testloader.dataset))
        test_acc = test_correct / (len(testloader.dataset)*16)       
 
        train_hist = np.histogram(np.array(train_out_dist), bins=10)
        wandb.log({'output distribution train': wandb.Histogram(np_histogram=train_hist)}, commit=False)
        test_hist = np.histogram(np.array(test_out_dist), bins=10)
        wandb.log({'output distribution test': wandb.Histogram(np_histogram=test_hist)}, commit=False)
        
        wandb.log({
            'Avg Train Loss': train_loss,
            'Train Accuracy': train_acc,
            'Avg Test Loss': test_loss,
            'Test Accuracy': test_acc, 
#            'Test Precision': test_pr,
#            'Test Recall': test_rec
            })
        
        if (epoch+1) % 100 == 0:
            torch.save(model.state_dict(), '/pylon5/hmcmutc/mdayao/saved_models/cross_val/goltsev_holdout_500_dropout_{}.h5'.format(epoch))
            wandb.save('/pylon5/hmcmutc/mdayao/saved_models/cross_val/goltsev_holdout_500_dropout_{}.h5'.format(epoch))
            
    return


if __name__ == '__main__':

    wandb.init(entity='mdayao', project='membnn')
    
    config = wandb.config
    config.batch_size = 32
    config.learning_rate = 0.01
    config.num_epochs = 500
    config.optimizer = 'SGD'
    device = torch.device('cuda')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--crossval', help = 'use crossval datasets', action='store_true')
    args = parser.parse_args()

    if not args.crossval:
        #raw_data_dir = '/hive/users/mdayao/hubmap/cytokit/Goltsev_mouse_spleen/data/20180101_codex_mouse_spleen_balbc_slide1/raw'
        raw_data_dir = '/hive/users/mdayao/hubmap/membrane_nn/data/raw_extracted'
        z_data_path = '/hive/users/mdayao/hubmap/membrane_nn/data/best_z.csv'
        label_path = '/hive/users/mdayao/hubmap/membrane_nn/data/class_labels.csv'
        
        trainset = MembraneImageDataset(label_path, z_data_path, raw_data_dir)
        testset = MembraneImageDataset(label_path, z_data_path, raw_data_dir, trainset=False)
        
        #image_shape = (1008, 1344)
        #image_shape = (256*3, 256*3)
        image_shape = (128, 128)
        
        model = SimpleCNN(image_shape)
        
        model.to(device)
        model.apply(weights_init)
        
        criterion = nn.BCELoss()
        if config.optimizer == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        elif config.optimizer == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr = config.learning_rate)
        
        trainloader = DataLoader(dataset=trainset, batch_size=config.batch_size, shuffle=True)
        testloader = DataLoader(dataset=testset, batch_size=config.batch_size, shuffle=False)
        
        wandb.watch(model, log='all')
        
        train_model(model, device, config.batch_size, trainloader, testloader, config.num_epochs, criterion, optimizer)
        
        torch.save(model.state_dict(), './saved_models/model_wavelet_500_patches_dropout_adam.h5')
        wandb.save('./saved_models/model_wavelet_500_patches_dropout_adam.h5')

    else:
        scratch_dir = '/pylon5/hmcmutc/mdayao/'
        #florida_dirs = [scratch_dir + 'florida/spleen_filtered', scratch_dir + 'florida/thymus_filtered']
        #florida_dirs = ['/hive/users/mdayao/hubmap/membrane_nn/florida/lymphnode_filtered', '/hive/users/mdayao/hubmap/membrane_nn/florida/thymus_filtered']
        #florida_dirs = [scratch_dir + 'florida/lymphnode_filtered', scratch_dir + 'florida/spleen_filtered']
        florida_dirs = [scratch_dir + 'florida/lymphnode_filtered', scratch_dir + 'florida/spleen_filtered', scratch_dir + 'florida/thymus_filtered']
        florida_labels = scratch_dir + 'florida/labels.csv'
        data_dir = scratch_dir + 'data/raw_extracted'
        label_path = scratch_dir + 'data/class_labels.csv'
        
        #labels = [label_path, florida_labels]
        labels = florida_labels
        #directories = [data_dir] + florida_dirs
        directories = florida_dirs
        #florida_sets = ['spleen', 'thymus']
        #florida_sets = ['lymphnode', 'thymus']
        #florida_sets = ['lymphnode', 'spleen']
        florida_sets = ['lymphnode', 'spleen', 'thymus']

        #test_label = florida_labels
        test_label = [label_path]
        #test_dir = [scratch_dir + 'florida/lymphnode_filtered']
        #test_set = ['lymphnode']
        #test_dir = ['/hive/users/mdayao/hubmap/membrane_nn/florida/spleen_filtered']
        #test_set = ['spleen']
        #test_dir = [scratch_dir + 'florida/thymus_filtered']
        #test_set = ['thymus']
        test_dir = [data_dir]
        test_set = ['goltsev']

        #trainset = CrossValMembraneImageDataset(labels, directories, florida_sets, trainset=True, all_florida=False)
        #testset = CrossValMembraneImageDataset(test_label, test_dir, test_set, trainset=False, all_florida=False)
        trainset = CrossValMembraneImageDataset(labels, directories, florida_sets, trainset=True, all_florida=True)
        testset = CrossValMembraneImageDataset(test_label, test_dir, test_set, trainset=False, all_florida=True)
        
        image_shape = (128, 128)
        
        model = SimpleCNN(image_shape)
        
        model.to(device)
        model.apply(weights_init)
    
        #model.load_state_dict(torch.load(scratch_dir + 'saved_models/cross_val/lymphnode_holdout_500_dropout_199.h5', map_location=device))
        #model.load_state_dict(torch.load(scratch_dir + 'saved_models/cross_val/thymus_holdout_500_dropout_399.h5', map_location=device))

        criterion = nn.BCELoss()
        if config.optimizer == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        elif config.optimizer == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr = config.learning_rate)
        
        trainloader = DataLoader(dataset=trainset, batch_size=config.batch_size, shuffle=True)
        testloader = DataLoader(dataset=testset, batch_size=config.batch_size, shuffle=False)
        
        wandb.watch(model, log='all')
        
        train_model(model, device, config.batch_size, trainloader, testloader, config.num_epochs, criterion, optimizer)
        
        torch.save(model.state_dict(), scratch_dir + '/saved_models/cross_val/goltsev_holdout_500_dropout.h5')
        wandb.save(scratch_dir + '/saved_models/cross_val/goltsev_holdout_500_dropout.h5')



