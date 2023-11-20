# Import dependencies
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision

#check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"running on: {device}")

batch_size = 512
epoch = 20
learning_rate = 0.001

#transform to tensors
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

#import mnist for trianing
train_dataset = torchvision.datasets.MNIST(root = 'torch_datasets',
                                           train = True,
                                           transform = transform,
                                           download = True)


#dataloader for training
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size = batch_size,
                                           shuffle = True)


#import mnist for testing
test_dataset = torchvision.datasets.MNIST(root= 'torch_datasets',
                                          train = False,
                                          transform = transform,
                                          download = True)

#dataloader for testing
test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size = 50,
                                              shuffle = False)

class Autoencoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        #encoder
        #input to 256
        self.encoder_dense_layer1 = nn.Linear(in_features = kwargs['input_shape'],
                                              out_features = 256)
        #256 to 128
        self.encoder_dense_layer2 = nn.Linear(in_features = 256,
                                              out_features = 128)
        
        #decoder
        # 128 to 256
        self.decoder_dense_layer1 = nn.Linear(in_features = 128,
                                              out_features = 256)
        #256 to input shape
        self.decoder_dense_layer2 = nn.Linear(in_features = 256,
                                              out_features = kwargs['input_shape'])
        
    def forward(self, data):
        x = data.reshape(data.shape[0], -1)
        x = torch.relu(self.encoder_dense_layer1(x))
        x = torch.relu(self.encoder_dense_layer2(x))
        x = torch.relu(self.decoder_dense_layer1(x))
        x = torch.relu(self.decoder_dense_layer2(x))
        x = x.reshape(data.shape)
        return x
    

class ConvAutoencoder(nn.Module):
    def __init__(self, **kwargs):
        super(ConvAutoencoder, self).__init__()
        #encoder
        #input_channel to 16
        self.encoder_conv1 = nn.Conv2d(kwargs['input_channel'], 16, 3 , padding=1)
        #16 to 4
        self.encoder_conv2 = nn.Conv2d(16, 4, 3, padding=1)
        self.encoder_pool = nn.MaxPool2d(2, 2)
        #4 to 8
        self.decoder_conv1 = nn.Conv2d(4, 8, 3, padding=1)
        #8 to 16
        self.decoder_conv2 = nn.Conv2d(8, 16, 3, padding=1)
        #16 to 1
        self.decoder_conv3 = nn.Conv2d(16, kwargs['input_channel'], 3, padding=1)

    def forward(self, x):
        x = torch.relu(self.encoder_conv1(x))
        x = self.encoder_pool(x)
        x = torch.relu(self.encoder_conv2(x))
        x = self.encoder_pool(x)
        x = torch.relu(self.decoder_conv1(x))
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = torch.sigmoid(self.decoder_conv2(x))
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = torch.sigmoid(self.decoder_conv3(x))
        return x
    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train(model, device, train_loader, optimizer, criterion, epochs, batch_size):
    #list to store losses
    losses = []
    # set model to train mode
    model.train()
    #training stage
    for epoch in range(epochs):
        loss = 0
        for data, _ in train_loader:
            optimizer.zero_grad() #resetting optimizer grads
            data = data.to(device)
            output = model(data)
            train_loss = criterion(output, data)
            train_loss.backward()
            optimizer.step()
            loss+= train_loss.item()
        #calculate epoch loss
        loss = loss/len(train_loader)
        losses.append(loss)
        print(f'epoch: {epoch + 1}/{epochs}, train_loss: {loss:.4f}')

def eval_plot(model, device, test_loader, criterion):
    #set model to evaluation mode
    model.eval()

    #create subplot
    fig = plt.figure(figsize=(15,5))

    #set torch.no_grad() to disable gradient computation and backprop
    with torch.no_grad():
        #evalue two instances for each class
        for n in range(2):
            #evaluate all classes
            for i in range(10):
                #take one class instance from the test dataset
                data = test_dataset.data[test_dataset.targets == i][[n]]
                data = data[:, None, :, :]
                data = data.float()
                reconstruction = model(data.to(device))
                #plot original image
                ax = plt.subplot(4, 10, i+1+n*20)
                plt.imshow(data.cpu().numpy().reshape(28,28))
                plt.gray()
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)

                #plot reconstrcuted image
                ax = plt.subplot(4, 10, i+11+n*20)
                plt.imshow(reconstruction.cpu().numpy().reshape(28,28))
                plt.gray()
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
    #show plot
    plt.show()

print('\n\nAutoencoder\n\n')

#select model
model = Autoencoder(input_shape=784).to(device)
#define optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#define loss criterion
criterion = nn.MSELoss()
#print model params
print(f'number of autoencoder params: {count_parameters(model)}')
#train model
train(model, device, train_loader, optimizer, criterion, epoch, batch_size)
#evaluate and plot results
eval_plot(model, device, test_loader, criterion)

print('\n\n CNN Autoencoder\n\n')

#select model
model = ConvAutoencoder(input_channel=1).to(device)
#define optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#define loss criterion
criterion = nn.MSELoss()
#print model paras
print(f'number of params of CNN autoencoder: {count_parameters(model)}')
#train model
train(model, device, train_loader, optimizer, criterion, epoch, batch_size)
#evaluate and plot results
eval_plot(model, device, test_loader, criterion)






