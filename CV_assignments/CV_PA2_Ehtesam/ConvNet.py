import time
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self, mode):
        super(ConvNet, self).__init__()
        
        # Define various layers here, such as in the tutorial example
        # self.conv1 = nn.Conv2D(...)
        
        # This will select the forward pass function based on mode for the ConvNet.
        # Based on the question, you have 5 modes available for step 1 to 5.
        # During creation of each ConvNet model, you will assign one of the valid mode.
        # This will fix the forward function (and the network graph) for the entire training/testing
        if mode == 1:
            self.fc1 = nn.Linear(784,100)
            self.fc_out = nn.Linear(100,10)
            self.forward = self.model_1
        elif mode == 2:
            self.conv1 = nn.Conv2d(in_channels= 1,
                                   out_channels= 40,
                                   kernel_size= (5,5),
                                   stride= (1,1))
            self.conv2 = nn.Conv2d(in_channels= 40,
                                   out_channels= 40,
                                   kernel_size= (5,5),
                                   stride= (1,1))
            self.pool = nn.MaxPool2d(kernel_size= (2,2),
                                     stride= (2,2))
            self.forward = self.model_2
        elif mode == 3:
            self.conv1 = nn.Conv2d(in_channels= 1,
                                   out_channels= 40,
                                   kernel_size= (5,5),
                                   stride= (1,1))
            self.conv2 = nn.Conv2d(in_channels= 40,
                                   out_channels= 40,
                                   kernel_size= (5,5),
                                   stride= (1,1))
            self.pool = nn.MaxPool2d(kernel_size= (2,2),
                                     stride= (2,2))
            self.fc1 = nn.Linear(640,100)
            self.fc_out = nn.Linear(100,10)
            self.forward = self.model_3
        elif mode == 4:
            self.conv1 = nn.Conv2d(in_channels= 1,
                                   out_channels= 40,
                                   kernel_size= (5,5),
                                   stride= (1,1))
            self.conv2 = nn.Conv2d(in_channels= 40,
                                   out_channels= 40,
                                   kernel_size= (5,5),
                                   stride= (1,1))
            self.pool = nn.MaxPool2d(kernel_size= (2,2),
                                     stride= (2,2))
            self.fc1 = nn.Linear(640,100)
            self.fc2 = nn.Linear(100,100)
            self.fc_out = nn.Linear(100,10)
            self.forward = self.model_4
        elif mode == 5:
            self.conv1 = nn.Conv2d(in_channels= 1,
                                   out_channels= 40,
                                   kernel_size= (5,5),
                                   stride= (1,1))
            self.conv2 = nn.Conv2d(in_channels = 40,
                                   out_channels= 40,
                                   kernel_size= (5,5),
                                   stride= (1,1))
            self.pool = nn.MaxPool2d(kernel_size= (2,2),
                                     stride= (2,2))
            self.fc1 - nn.Linear(640,1000)
            self.fc2 = nn.Linear(1000,1000)
            self.fc_out = nn.Linear(1000,10)
            self.forward = self.model_5
        else: 
            print("Invalid mode ", mode, "selected. Select between 1-5")
            exit(0)
        
        
    # Baseline model. step 1
    def model_1(self, X):
        # ======================================================================
        # One fully connected layer.
        X = X.reshape(X.shape[0], -1)
        X = torch.sigmoid(self.fc1(X))
        fc1 = self.fc_out(X)
        return fc1
        # Uncomment the following return stmt once method implementation is done.
        # return  fcl
        # Delete line return NotImplementedError() once method is implemented.
        #return NotImplementedError()

    # Use two convolutional layers.
    def model_2(self, X):
        # ======================================================================
        X = torch.sigmoid(self.conv1(X))
        X = self.pool(X)
        X = torch.sigmoid(self.conv2(X))
        X = self.pool(X)
        X = X.reshape(X.shape[0], -1)
        X = torch.sigmoid(self.fc1(X))
        fc1 = self.fc_out(X)
        return fc1
        # Uncomment the following return stmt once method implementation is done.
        # return  fcl
        # Delete line return NotImplementedError() once method is implemented.

        #return NotImplementedError()

    # Replace sigmoid with ReLU.
    def model_3(self, X):
        # ======================================================================
        # Two convolutional layers + one fully connected layer, with ReLU.
        X = F.relu(self.conv1(X))
        X = self.pool(X)
        X = F.relu(self.conv2(X))
        X = self.pool(X)
        X = X.reshape(X.shape[0], -1)
        X = F.relu(self.fc1(X))
        fc1 = self.fc_out(X)
        return fc1
        # Delete line return NotImplementedError() once method is implemented.
        #return NotImplementedError()

    # Add one extra fully connected layer.
    def model_4(self, X):
        # ======================================================================
        # Two convolutional layers + two fully connected layers, with ReLU.
        X = F.relu(self.conv1(X))
        X = self.pool(X)
        X = F.relu(self.conv2(X))
        X = self.pool(X)
        X = X.reshape(X.shape[0], -1)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        fc1 = self.fc_out(X)
        return fc1
        # Delete line return NotImplementedError() once method is implemented.
        #return NotImplementedError()

    # Use Dropout now.
    def model_5(self, X):
        # ======================================================================
        # Two convolutional layers + two fully connected layers, with ReLU.
        # and  + Dropout.
        X = F.relu(self.conv1(X))
        X = self.pool(X)
        X = F.relu(self.conv2(X))
        X = self.pool(X)
        X = X.reshape(X.shape[0], -1)
        X = self.dropout(X)
        X = F.relu(self.fc1(X))
        X = self.dropout(X)
        X = F.relu(self.fc2(X))
        X = self.dropout(X)
        fc1 = self.fc_out(X)
        return fc1
        # Uncomment the following return stmt once method implementation is done.
        # return  fcl
        # Delete line return NotImplementedError() once method is implemented.
        #return NotImplementedError()
    
    
