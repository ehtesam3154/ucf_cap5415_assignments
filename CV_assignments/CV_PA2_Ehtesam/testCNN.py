from __future__ import print_function
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from ConvNet import ConvNet
import argparse
import numpy as np 
import sys
from tee import StdoutTee
import matplotlib.pyplot as plt

# def custom_print(message_to_print, log_file='output.txt'):
#     print(message_to_print)
#     with open(log_file, 'a') as of:
#         of.write(message_to_print + '\n')

def train(model, device, train_loader, optimizer, criterion, epoch, batch_size):
    '''
    Trains the model for an epoch and optimizes it.
    model: The model to train. Should already be in correct device.
    device: 'cuda' or 'cpu'.
    train_loader: dataloader for training samples.
    optimizer: optimizer to use for model parameter updates.
    criterion: used to compute loss for prediction and target 
    epoch: Current epoch to train for.
    batch_size: Batch size to be used.
    '''
    
    # Set model to train mode before each epoch
    model.train()
    
    # Empty list to store losses 
    losses = []
    correct = 0
    
    # Iterate over entire training samples (1 epoch)
    for batch_idx, batch_sample in enumerate(train_loader):
        data, target = batch_sample
        
        # Push data/label to correct device
        data, target = data.to(device), target.to(device)
        
        # Reset optimizer gradients. Avoids grad accumulation (accumulation used in RNN).
        optimizer.zero_grad()
        
        # Do forward pass for current set of data
        output = model(data)
        
        # ======================================================================
        # Compute loss based on criterion
        loss = criterion(output, target)
        # Remove NotImplementedError and assign correct loss function.
        #loss = NotImplementedError()
        
        # Computes gradient based on final loss
        loss.backward()
        
        # Store loss
        losses.append(loss.item())
        
        # Optimize model parameters based on learning rate and gradient 
        optimizer.step()
        
        # Get predicted index by selecting maximum log-probability
        pred = output.argmax(dim=1, keepdim=False) #keepdim has to be False otherwise accuracy goes above 100%
        
        # ======================================================================
        # Count correct predictions overall 
        correct += (pred==target).sum()
        # Remove NotImplementedError and assign counting function for correct predictions.
        #correct = NotImplementedError()
        
    train_loss = float(np.mean(losses))
    train_accuracy = 100. * correct / len(train_loader.dataset)

    print('Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        train_loss, correct, len(train_loader.dataset), train_accuracy))
    # print('Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #     float(np.mean(losses)), correct, (batch_idx+1) * batch_size,
    #     100. * correct / ((batch_idx+1) * batch_size)))
    # train_accuracy = 100. * correct / len(train_loader.dataset)

    # print('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #     train_loss, correct, len(train_loader.dataset), train_accuracy))
    return train_loss, train_accuracy
    


def test(model, device, test_loader, criterion):
    '''
    Tests the model.
    model: The model to train. Should already be in correct device.
    device: 'cuda' or 'cpu'.
    test_loader: dataloader for test samples.
    '''
    
    # Set model to eval mode to notify all layers.
    model.eval()
    
    losses = []
    correct = 0
    
    # Set torch.no_grad() to disable gradient computation and backpropagation
    with torch.no_grad():
        for batch_idx, sample in enumerate(test_loader):
            data, target = sample
            data, target = data.to(device), target.to(device)
            

            # Predict for data by doing forward pass
            output = model(data)
            
            # ======================================================================
            loss = criterion(output, target)
            # Remove NotImplementedError and assign correct loss function.
            # Compute loss based on same criterion as training 
            #loss = NotImplementedError()
            
            # Append loss to overall test loss
            losses.append(loss.item())
            
            # Get predicted index by selecting maximum log-probability
            pred = output.argmax(dim=1, keepdim=False)
            
            # ======================================================================
            # Count correct predictions overall
            correct += (pred==target).sum()
            # Remove NotImplementedError and assign counting function for correct predictions.
            #correct = NotImplementedError()

    test_loss = float(np.mean(losses))
    accuracy = 100. * correct / len(test_loader.dataset)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset), accuracy))
    
    return test_loss, accuracy
    

def run_main(FLAGS):
    # Check if cuda is available
    use_cuda = torch.cuda.is_available()
    
    # Set proper device based on cuda availability 
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Torch device selected: ", device)
    
    # Initialize the model and send to device 
    model = ConvNet(FLAGS.mode).to(device)
    
    # ======================================================================
    # Define loss function.
    criterion = nn.CrossEntropyLoss()
    # Remove NotImplementedError and assign correct loss function.
    #criterion = NotImplementedError()
    
    # ======================================================================
    optimizer = optim.SGD(model.parameters(), lr=FLAGS.learning_rate)
    # Remove NotImplementedError and assign appropriate optimizer with learning rate and other paramters.
    #optimizer = NotImplementedError()
        
    
    # Create transformations to apply to each data sample 
    # Can specify variations such as image flip, color flip, random crop, ...
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    # Load datasets for training and testing
    # Inbuilt datasets available in torchvision (check documentation online)
    dataset1 = datasets.MNIST('./data/', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('./data/', train=False,
                       transform=transform)
    train_loader = DataLoader(dataset1, batch_size = FLAGS.batch_size, 
                                shuffle=True, num_workers=4)
    test_loader = DataLoader(dataset2, batch_size = FLAGS.batch_size, 
                                shuffle=False, num_workers=4)
    
    best_accuracy = 0.0
    train_losses = []  # To store training losses
    test_losses = []   # To store testing losses
    
    # Run training for n_epochs specified in config 
    for epoch in range(1, FLAGS.num_epochs + 1):
        print(f"Epoch {epoch} " + '-'*60)
        train_loss, train_accuracy = train(model, device, train_loader,
                                            optimizer, criterion, epoch, FLAGS.batch_size)
        test_loss, test_accuracy = test(model, device, test_loader, criterion)
        
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy

        # Append losses to the lists
        train_losses.append(train_loss)
        test_losses.append(test_loss)
    
    
    print("best accuracy is {:2.2f}".format(best_accuracy))
    
    print("Training and evaluation finished!")

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, FLAGS.num_epochs + 1), train_losses, label="Training Loss")
    plt.plot(range(1, FLAGS.num_epochs + 1), test_losses, label="Testing Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Testing Loss Curves")
    plt.grid()
    plt.savefig('loss_curves.png')
    plt.close()

    # Restore the original stdout
    sys.stdout = original_stdout
  

if __name__ == '__main__':
    # Set parameters for Sparse Autoencoder
    parser = argparse.ArgumentParser('CNN Exercise.')
    parser.add_argument('--mode',
                        type=int, default=1,
                        help='Select mode between 1-5.')
    parser.add_argument('--learning_rate',
                        type=float, default=0.1,
                        help='Initial learning rate.')
    parser.add_argument('--num_epochs',
                        type=int,
                        default=60,
                        help='Number of epochs to run trainer.')
    parser.add_argument('--batch_size',
                        type=int, default=10,
                        help='Batch size. Must divide evenly into the dataset sizes.')
    parser.add_argument('--log_dir',
                        type=str,
                        default='logs',
                        help='Directory to put logging.')
    
    FLAGS = None
    FLAGS, unparsed = parser.parse_known_args()

    output_file = 'output.txt'

    # Open a file for writing
    with open(output_file, 'w') as file:
        # Redirect stdout to both the terminal and the file
        original_stdout = sys.stdout
        sys.stdout = file

        run_main(FLAGS)