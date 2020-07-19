## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I

# helper conv() function to set up a convolutional 2D layer with an optional attached batch norm layer
def conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True, batch_norm=True):
    """Creates a 2D convolutional layer (for downscaling width and height of the input tensor) with an 
       attached optional batch normalization layer.
       
       Arguments:
       
       in_channels:  input channels resp. depth of input tensor
       out_channels: output channels resp. depth of output tensor
       kernel_size:  kernal size of transposed convolutional filter (default: 3)
       stride:       stride to shift the filter kernel along tensor width and height (default: 1)
       padding:      number of rows / colums padded with zeros on the outer rims of the tensor (default: 1)
       bias:         bias (default: True)
       batch_norm:   flag to switch batch normalization on (batch_norm = True) or off (batch_norm = False)
    """
    
    # initialize list of layers
    layers = []
    
    # specify 2D convolutional layer
    conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias)
    
    # append 2D convolutional layer
    layers.append(conv_layer)
    
    if batch_norm:
        # append 2D batch normalization layer
        layers.append(nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
    
    # return sequential stack of layers
    return nn.Sequential(*layers)


# helper lin() function to set up a linear 1D layer with an optional attached batch norm layer
def lin(in_features, out_features, bias=True, batch_norm=True):
    """Creates a 2D convolutional layer (for downscaling width and height of the input tensor) with an 
       attached optional batch normalization layer.
       
       Arguments:
       
       in_features:  input features of input tensor
       out_features: output features of output tensor
       bias:         bias (default: True)
       batch_norm:   flag to switch batch normalization on (batch_norm = True) or off (batch_norm = False)
    """
    
    # initialize list of layers
    layers = []
    
    # specify 1D linear layer
    lin_layer = nn.Linear(in_features, out_features, bias)
    
    # append 1D linear layer
    layers.append(lin_layer)
    
    if batch_norm:
        # append 1D batch normalization layer
        layers.append(nn.BatchNorm1d(out_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
    
    # return sequential stack of layers
    return nn.Sequential(*layers)


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        # self.conv1 = nn.Conv2d(1, 32, 5)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or 
        # batch normalization) to avoid overfitting
                
        # set basic depth of sequential convolutional layers
        conv_dim = 32
        
        ## Define layers of a CNN
        
        ## Feature extractor
        
        # 1st convolutional layer with 1 x 3 x 3 filter kernel (sees a 1 x 224 x 224 tensor)
        self.conv1 = conv(in_channels=1, out_channels=conv_dim, 
                          kernel_size=3, stride=1, padding=1, bias=True, batch_norm=True)
                
        # 2nd convolutional layer with 32 x 3 x 3 filter kernel (sees a 32 x 112 x 112 tensor)
        self.conv2 = conv(in_channels=conv_dim, out_channels=2*conv_dim, 
                          kernel_size=3, stride=2, padding=1, bias=True, batch_norm=True)
                
        # 3rd convolutional layer with 64 x 3 x 3 filter kernel (sees 64 x 28 x 28 tensor)
        self.conv3 = conv(in_channels=2*conv_dim, out_channels=4*conv_dim, 
                          kernel_size=3, stride=2, padding=1, bias=True, batch_norm=True)
        
        # dropout layer (p=0.2)
        self.drop = nn.Dropout(p=0.2)
        
        # Max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        ## Linear Classifier
        
        # 1st fully-connected linear layer 1 with 1024 nodes (sees a 128 x 7 x 7 tensor)
        self.fc1 = lin(in_features=128*7*7, out_features=1024, bias=True, batch_norm=True)
                        
        # 2nd and final fully-connected linear layer 2 with 68 x 2 = 136 nodes (sees a 1 x 1024 tensor)
        self.fc2 = lin(in_features=1024, out_features=136, bias=True, batch_norm=False)

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        ## Feature extractor
        
        # Convolutional hidden layer 1 with batch normalization, relu activation function and max pooling
        x = self.pool(F.relu(self.conv1(x)))
        
        # Dropout layer 1
        x = self.drop(x)
        
        # Convolutional hidden layer 2 with batch normalization, relu activation function and max pooling
        x = self.pool(F.relu(self.conv2(x)))
        
        # Dropout layer 2
        x = self.drop(x)
        
        # Convolutional hidden layer 3 with batch normalization, relu activation function and max pooling
        x = self.pool(F.relu(self.conv3(x)))
        
        # Dropout layer 3
        x = self.drop(x)
                        
        ## Classifier
        
        # Flatten 128 x 7 x 7 input tensor to first fully conntected layer
        x = x.view(x.size(0), -1)
        
        # Fully connected hidden layer fc1 with batch normalization and relu activation function
        x = F.relu(self.fc1(x))
                
        # Dropout layer 4
        x = self.drop(x)
                
        # Fully connected hidden layer fc2 (no batch normalization, no activation function) => return 
        # facial keypoint coordinates in (x, y) pairs
        x = self.fc2(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
    
    def predict(self, x):
        
        # Predict outputs in forward pass (without dropout) while also returning activations and feature maps
        
        # Initialize dictionary of activations, feature maps and layer outputs
        activations = {}
        feature_maps = {}
        layer_outputs = {}
        
        ## Feature extractor
        
        # Activations with batch normalization of convolutional hidden layer conv1
        a = self.conv1(x)
        activations['conv1'] = a
        
        # Feature map after applying relu activation function on activations of convolutional layer conv1
        h = F.relu(a)
        feature_maps['conv1'] = h
        
        # Max pooling of feature maps of convolutional layer conv1
        out = self.pool(h)
        layer_outputs['pool_conv1'] = out
        
        # Dropout layer 1
        x = self.drop(out)
        
        # Activations with batch normalization of convolutional hidden layer conv2
        a = self.conv2(x)
        activations['conv2'] = a
        
        # Feature map after applying relu activation function on activations of convolutional layer conv2
        h = F.relu(a)
        feature_maps['conv2'] = h
        
        # Max pooling of feature maps of convolutional layer conv2
        out = self.pool(h)
        layer_outputs['pool_conv2'] = out
        
        # Dropout layer 2
        x = self.drop(out)
        
        # Activations with batch normalization of convolutional hidden layer conv3
        a = self.conv3(x)
        activations['conv3'] = a
        
        # Feature map after applying relu activation function on activations of convolutional layer conv3
        h = F.relu(a)
        feature_maps['conv3'] = h
        
        # Max pooling of feature maps of convolutional layer conv3
        out = self.pool(h)
        layer_outputs['pool_conv3'] = out
        
        # Dropout layer 3
        x = self.drop(out)
                        
        ## Classifier
        
        # Flatten 128 x 7 x 7 input tensor to first fully conntected layer
        x = x.view(x.size(0), -1)
        
        # Activations with batch normalization of fully connected layer fc1
        a = self.fc1(x)
        activations['fc1'] = a
        
        # Layer output after applying relu activation function on activations of fully connected layer fc1
        out = F.relu(a)
        layer_outputs['fc1'] = out
        
        # Dropout layer 4
        x = self.drop(out)
                
        # Add fully connected hidden layer fc2 (no batch normalization, no activation function) => return 
        # facial keypoint coordinates in (x, y) pairs
        key_pts = self.fc2(x)
        
        # Return predictions (tensor) plus activations, feature maps and layer outputs (dictionary of tensors)
        return key_pts, activations, feature_maps, layer_outputs
        # return key_pts, feature_maps
