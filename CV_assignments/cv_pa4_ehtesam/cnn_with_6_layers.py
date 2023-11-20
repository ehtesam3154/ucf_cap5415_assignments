# Import dependencies
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'running on {device}', end = '\n')

# define CNN with 6 layers
class CNN(nn.Module):
    def __init__(self, **kwargs) -> None:
        super(CNN, self).__init__()

        self.in_shape = kwargs['input_shape']
        self.in_channel = kwargs['input_shape'][1]
        self.n_classes = kwargs['n_classes']

        #conv layers
        self.conv1 = nn.Conv2d(self.in_channel, 16, 3, padding=1) # in_channel -> 16
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1) # 16-> 32
        self.conv3 = nn.Conv2d(32, 32, 3, padding=1) # 32-> 64
        self.conv4 = nn.Conv2d(32, 64, 3, padding=1) #64-> 128
        self.conv5 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv6 = nn.Conv2d(64, 128, 3, padding=1)  
        self.pool = nn.MaxPool2d(2,2) #maxpooling layer

        #fully connected layers
        self.fc1 = nn.Linear(self._fc_units(), 1000) #flattened_cnn -> 1000
        self.fc2 = nn.Linear(1000, 1000)
        self.fc3 = nn.Linear(1000, self.n_classes)

    def forward(self, x):
        x = self._forward_conv(x)
        x = self._forward_fc(x)
        x = F.log_softmax(self.fc3(x), dim=1)
        return x

    def _forward_conv(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        # x = self.pool(x)
        x = torch.relu(self.conv3(x))
        x = self.pool(x)
        x = torch.relu(self.conv4(x))
        # x = self.pool(x)
        x = torch.relu(self.conv5(x))
        x = self.pool(x)
        x = torch.relu(self.conv6(x))
        x = self.pool(x)
        return x
    
    def _forward_fc(self, x):
        x = x.reshape(x.shape[0], -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return x
    
    def _fc_units(self):
        dummy_input = torch.rand(1, self.in_channel, *self.in_shape[2:])
        x = self._forward_conv(dummy_input)
        x = x.view(x.size(0), -1)
        return x.size(1)

    
def train(model, device, train_loader, test_loader, optimizer, criterion, epochs, batch_size):
    #list to store losses
    log = {'train_loss': [],
           'val_loss': [],
           'train_acc': [],
           'val_acc': []}
    
    #training model
    for epoch in range(epochs):
        print(f'epoch: {epoch+1}/{epochs}', end='\t')
        #set model to training mode
        model.train()
        train_loss, correct = 0, 0
        for inputs, labels in train_loader:
            #all standard training procedure
            optimizer.zero_grad()
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            #get predicted class with max log probability
            preds = outputs.argmax(dim=1, keepdim=False)
            correct += (preds==labels).sum() #count number of correct preds

        #calculate epoch train accuracy and loss
        train_accuracy = correct/len(train_loader.dataset) * 100
        train_loss = train_loss/len(train_loader)

        #store epoch loss and accuracy
        log['train_loss'].append(train_loss)
        log['train_acc'].append(train_accuracy)
        print(f'train loss: {train_loss:.4f}\t train accuracy: {train_accuracy:.4f}', end='\n')

        #set model to evaluation mode
        model.eval()
        val_loss, correct = 0, 0
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            val_loss += criterion(outputs, labels).item()
            #get predictions and count correct ones
            preds = outputs.argmax(dim=1, keepdim=False)
            correct += (preds==labels).sum()

        val_accuracy = correct/len(test_loader.dataset) * 100

        #calculate epoch validation loss and accuracy
        val_loss = val_loss/len(test_loader)
        log['val_loss'].append(val_loss)
        log['val_acc'].append(val_accuracy)
        print(f'validation loss: {val_loss:.4f}\t validation accuracy: {val_accuracy:.4f}', end='\n')

    return log

def plot_loss_accuracy(log, epoch):
    metrics = ['loss', 'acc']
    colors = ['tab:blue', 'tab:orange']

    #create subplots
    fig, ax = plt.subplots(1,2, figsize=(12,5))

    #set figure title
    fig.suptitle(f'MNIST dataset performance (n_epoch={epoch})', fontsize=14, y=1.05)

    #set fig color to white
    fig.set_facecolor('white')

    for i in range(2):
        metric_name = metrics[i]
        #plot train and test metric
        ax[i].plot(log[f'train_{metric_name}'], color = colors[0], linewidth = 1.5, label= f'Train {metric_name.capitalize()}')
        ax[i].plot(log[f'val_{metric_name}'], color = colors[1], linewidth = 1.5, label = f'Test {metric_name.capitalize()}')

        #set title for the subplot
        ax[i].set_title(f'{metric_name.capitalize()} vs epoch', size=12, y=1.01)
        #set labels for x and y axis
        ax[i].set_xlabel('epoch', size=10)
        ax[i].set_ylabel(metric_name.capitalize(), size=10)
        #show plot legends
        ax[i].legend()
        #turn on grid
        ax[i].grid(linestyle='dashed')

    plt.show()

if __name__ == '__main__':
    batch_size = 32
    epoch = 30
    learning_rate = 0.001

    #create transformation to apply to each data sample
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    #import cifar10 dataset for training
    train_dataset = torchvision.datasets.CIFAR10(root= 'torch_datasets',
                                                 train = True,
                                                 transform=transform,
                                                 download=True)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=False)
    
    #import cifar10 for testing
    test_dataset = torchvision.datasets.CIFAR10(root= 'torch_datasets',
                                                 train = False,
                                                 transform=transform,
                                                 download=True)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                               batch_size=batch_size,
                                               shuffle=False)
    
    n_classes = len(train_dataset.classes)

    #initialize model
    model = CNN(input_shape = [1,3,32,32], n_classes=n_classes).to(device)
    #define optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    #define loss criterion
    criterion = nn.NLLLoss()

    #train model
    log = train(model, device, train_loader, test_loader,
                optimizer, criterion, epoch, batch_size)
    
    #plot results
    plot_loss_accuracy(log, epoch)



        







