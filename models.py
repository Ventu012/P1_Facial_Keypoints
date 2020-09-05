import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        ## self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)
        self.conv1 = nn.Conv2d(1, 32, 5)
        # output size = (W-F)/S +1 = (224-5)/1 +1 = 220
        # the output Tensor for one image, will have the dimensions: (32, 220, 220)
        
        # Initialize the weights by performing Xavier initialization
        nn.init.xavier_normal_(self.conv1.weight)
        
        # maxpool layer
        # pool with kernel_size=2, stride=2
        self.pool1 = nn.MaxPool2d(2, 2)
        # 220/2 = 110
        # the output Tensor for one image, will have the dimensions: (32, 110, 110)
        
        ## self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)
        self.conv2 = nn.Conv2d(32, 64, 3)
        # output size = (W-F)/S +1 = (110-3)/1 + 1 = 108
        # the output Tensor for one image, will have the dimensions: (64, 108, 108)
        
        # Initialize the weights by performing Xavier initialization
        nn.init.xavier_normal_(self.conv2.weight)
        
        # maxpool layer
        # pool with kernel_size=2, stride=2
        self.pool2 = nn.MaxPool2d(2, 2)
        # 108/2 = 54
        # the output Tensor for one image, will have the dimensions: (64, 54, 54)
        
        ## self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)
        self.conv3 = nn.Conv2d(64, 128, 3)
        # output size = (W-F)/S +1 = (54-3)/1 + 1 = 52
        # the output Tensor for one image, will have the dimensions: (128, 52, 52)
        
        # Initialize the weights by performing Xavier initialization
        nn.init.xavier_normal_(self.conv3.weight)
        
        # maxpool layer
        # pool with kernel_size=2, stride=2
        self.pool3 = nn.MaxPool2d(2, 2)
        # 52/2 = 26
        # the output Tensor for one image, will have the dimensions: (128, 26, 26)
        
        ## self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)
        self.conv4 = nn.Conv2d(128, 256, 3)
        # output size = (W-F)/S +1 = (26-3)/1 + 1 = 24
        # the output Tensor for one image, will have the dimensions: (256, 24, 24)
        
        # Initialize the weights by performing Xavier initialization
        nn.init.xavier_normal_(self.conv4.weight)
        
        # maxpool layer
        # pool with kernel_size=2, stride=2
        self.pool4 = nn.MaxPool2d(2, 2)
        # 24/2 = 12
        # the output Tensor for one image, will have the dimensions: (256, 12, 12)
        
        ## self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)
        self.conv5 = nn.Conv2d(256, 512, 3)
        # output size = (W-F)/S +1 = (12-3)/1 + 1 = 10
        # the output Tensor for one image, will have the dimensions: (512, 10, 10)
        
        # Initialize the weights by performing Xavier initialization
        nn.init.xavier_normal_(self.conv5.weight)
        
        # maxpool layer
        # pool with kernel_size=2, stride=2
        #self.pool5 = nn.MaxPool2d(2, 2)
        # 10/2 = 5
        # the output Tensor for one image, will have the dimensions: (512, 5, 5)
        
        # Fully-connected (linear) layers
        self.fc1 = nn.Linear(512*10*10, 1024)
        
        # Initialize the weights by performing Xavier initialization
        nn.init.xavier_normal_(self.fc1.weight)
        
        self.fc2 = nn.Linear(1024, 512)
        
        # Initialize the weights by performing Xavier initialization
        nn.init.xavier_normal_(self.fc2.weight)
        
        self.fc3 = nn.Linear(512, 68*2)
        
        # Initialize the weights by performing Xavier initialization
        nn.init.xavier_normal_(self.fc3.weight)
        
        # Dropout
        self.dropout = nn.Dropout(p=0.25)

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        
        # 5 conv/relu + pool layers
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.pool4(F.relu(self.conv4(x)))
        # x = self.pool5(F.relu(self.conv5(x)))
        x = F.relu(self.conv5(x))
        
        # Prep for linear layer / Flatten
        x = x.view(x.size(0), -1)
        
        # linear layers with dropout in between
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x