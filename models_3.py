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
        
        # 1st dropout layer (p=0.1)
        self.drop1 = nn.Dropout(p=0.1)
        
        # 2nd convolutional layer with 32 x 3 x 3 filter kernel (sees a 32 x 112 x 112 tensor)
        self.conv2 = conv(in_channels=conv_dim, out_channels=2*conv_dim, 
                          kernel_size=3, stride=1, padding=1, bias=True, batch_norm=True)
        
        # 2nd dropout layer (p=0.15)
        self.drop2 = nn.Dropout(p=0.15)
        
        # 3rd convolutional layer with 64 x 3 x 3 filter kernel (sees 64 x 56 x 56 tensor)
        self.conv3 = conv(in_channels=2*conv_dim, out_channels=4*conv_dim, 
                          kernel_size=3, stride=1, padding=1, bias=True, batch_norm=True)
        
        # 3rd dropout layer (p=0.2)
        self.drop3 = nn.Dropout(p=0.2)
        
        # Fourth convolutional layer with 128 x 3 x 3 filter kernel (sees 128 x 28 x 28 tensor)
        self.conv4 = conv(in_channels=4*conv_dim, out_channels=8*conv_dim, 
                          kernel_size=3, stride=1, padding=1, bias=True, batch_norm=True)
        
        # 4th dropout layer (p=0.25)
        self.drop4 = nn.Dropout(p=0.25)
        
        # Fifth convolutional layer with 256 x 3 x 3 filter kernel (sees 256 x 14 x 14 tensor)
        self.conv5 = conv(in_channels=8*conv_dim, out_channels=16*conv_dim, 
                          kernel_size=3, stride=1, padding=1, bias=True, batch_norm=True)
        
        # 5th dropout layer (p=0.3)
        self.drop5 = nn.Dropout(p=0.3)
        
        # Max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        ## Linear Classifier
        
        # 1st fully-connected linear layer 1 with 1024 nodes (sees a 512 x 7 x 7 tensor)
        self.fc1 = lin(in_features=512 * 7 * 7, out_features=1024, bias=True, batch_norm=True)
        
        # 6th dropout layer (p=0.3)
        self.drop6 = nn.Dropout(p=0.3)
        
        # 2nd fully-connected linear layer 2 with 1024 nodes (sees a 1 * 1024 tensor)
        self.fc2 = lin(in_features=1024, out_features=1024, bias=True, batch_norm=True)
        
        # 7th dropout layer (p=0.3)
        self.drop7 = nn.Dropout(p=0.3)
        
        # 3rd and final fully-connected linear layer 3 with 68 x 2 = 136 nodes (sees a 1 x 1024 tensor)
        self.fc3 = lin(in_features=1024, out_features=136, bias=True, batch_norm=False)

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        ## Feature extractor
        
        # Convolutional layer 1 with batch normalization, elu activation function and max pooling
        x = self.pool(F.elu(self.conv1(x), alpha=0.1)) # default alpha = 1.0
        
        # Dropout layer 1
        x = self.drop1(x)
        
        # Convolutional layer 2 with batch normalization, elu activation function and max pooling
        x = self.pool(F.elu(self.conv2(x), alpha=0.1)) # default alpha = 1.0
        
        # Dropout layer 2
        x = self.drop2(x)
        
        # Convolutional layer 3 with batch normalization, elu activation function and max pooling
        x = self.pool(F.elu(self.conv3(x), alpha=0.1)) # default alpha = 1.0
        
        # Dropout layer 3
        x = self.drop3(x)
        
        # Convolutional layer 4 with batch normalization, elu activation function and max pooling
        x = self.pool(F.elu(self.conv4(x), alpha=0.1)) # default alpha = 1.0
        
        # Dropout layer 4
        x = self.drop4(x)
        
        # Convolutional layer 5 with batch normalization, elu activation function and max pooling
        x = self.pool(F.elu(self.conv5(x), alpha=0.1)) # default alpha = 1.0
        
        # Dropout layer 5
        x = self.drop5(x)
        
        ## Classifier
        
        # Flatten 512 x 7 x 7 input tensor to first fully conntected layer
        x = x.view(x.size(0), -1)
        
        # Fully connected layer 1 with batch normalization and leaky relu activation function
        x = F.leaky_relu(self.fc1(x), negative_slope=0.01)
        
        # Dropout layer 6
        x = self.drop6(x)
        
        # Fully connected layer 2 with batch normalization and leaky relu activation function
        x = F.leaky_relu(self.fc2(x), negative_slope=0.01)
        
        # Dropout layer 7
        x = self.drop7(x)
        
        # Fully connected layer 3 (no batch normalization, no activation function) => return 
        # facial keypoint coordinates in (x, y) pairs
        x = self.fc3(x)
        
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
        
        # Feature map after applying elu activation function on activations of convolutional layer conv1
        h = F.elu(a, alpha=0.1) # default alpha = 1.0
        feature_maps['conv1'] = h
        
        # Max pooling of feature maps of convolutional layer conv1
        out = self.pool(h)
        layer_outputs['pool_conv1'] = out
        
        # Dropout layer 1
        x = self.drop1(out)
        
        # Activations with batch normalization of convolutional layer conv2
        a = self.conv2(x)
        activations['conv2'] = a
        
        # Feature map after applying elu activation function on activations of convolutional layer conv2
        h = F.elu(a, alpha=0.1) # default alpha = 1.0
        feature_maps['conv2'] = h
        
        # Max pooling of feature maps of convolutional layer conv2
        out = self.pool(h)
        layer_outputs['pool_conv2'] = out
        
        # Dropout layer 2
        x = self.drop2(out)
        
        # Activations with batch normalization of convolutional layer conv3
        a = self.conv3(x)
        activations['conv3'] = a
        
        # Feature map after applying elu activation function on activations of convolutional layer conv3
        h = F.elu(a, alpha=0.1) # default alpha = 1.0
        feature_maps['conv3'] = h
        
        # Max pooling of feature maps of convolutional layer conv3
        out = self.pool(h)
        layer_outputs['pool_conv3'] = out
        
        # Dropout layer 3
        x = self.drop3(out)
        
        # Activations with batch normalization of convolutional layer conv4
        a = self.conv4(x)
        activations['conv4'] = a
        
        # Feature map after applying elu activation function on activations of convolutional layer conv4
        h = F.elu(a, alpha=0.1) # default alpha = 1.0
        feature_maps['conv4'] = h
        
        # Max pooling of feature maps of convolutional layer conv4
        out = self.pool(h)
        layer_outputs['pool_conv4'] = out
        
        # Dropout layer 4
        x = self.drop4(out)
        
        # Activations with batch normalization of convolutional layer conv5
        a = self.conv5(x)
        activations['conv5'] = a
        
        # Feature map after applying elu activation function on activations of convolutional layer conv5
        h = F.elu(a, alpha=0.1) # default alpha = 1.0
        feature_maps['conv5'] = h
        
        # Max pooling of feature maps of convolutional layer conv5
        out = self.pool(h)
        layer_outputs['pool_conv5'] = out
        
        # Dropout layer 5
        x = self.drop5(out)
        
        ## Classifier
        
        # Flatten 512 x 7 x 7 input tensor to first fully conntected layer
        x = x.view(x.size(0), -1)
        
        # Activations with batch normalization of fully connected layer fc1
        a = self.fc1(x)
        activations['fc1'] = a
        
        # Layer output after applying leaky_relu activation function on activations of fully connected layer fc1
        out = F.leaky_relu(a, negative_slope=0.01)
        layer_outputs['fc1'] = out
        
        # Add dropout layer 6
        x = self.drop6(out)
        
        # Activations with batch normalization of fully connected layer fc1
        a = self.fc2(x)
        activations['fc2'] = a
        
        # Layer output after applying leaky_relu activation function on activations of fully connected layer fc2
        out = F.leaky_relu(a, negative_slope=0.01)
        layer_outputs['fc2'] = out
        
        # Dropout layer 7
        x = self.drop7(out)
        
        # Add fully connected hidden layer 3 (no batch normalization, no activation function) => return 
        # facial keypoint coordinates in (x, y) pairs
        key_pts = self.fc3(x)
        
        # Return predictions (tensor) plus activations, feature maps and layer outputs (dictionary of tensors)
        return key_pts, activations, feature_maps, layer_outputs