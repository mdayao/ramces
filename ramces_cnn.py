import torch
import torch.nn as nn

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
        self.sigmoid = nn.Sigmoid()
        

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out.view(-1, self.last_layer_size)
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.sigmoid(out)
        return out
