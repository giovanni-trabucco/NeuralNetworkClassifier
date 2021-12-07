#!/usr/bin/env python
# coding: utf-8

import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split 
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from networkz import NonConvolutionalNet, ConvolutionalNet

train_dataset = torchvision.datasets.FashionMNIST('classifier_data', train=True, download=True)
test_dataset  = torchvision.datasets.FashionMNIST('classifier_data', train=False, download=True)


# ### Setting up Dataset and DataLoader
# I want to build my own custom dataset to convert from PIL to Tensors



class CustomDataset(Dataset):
    def __init__(self, list_of_pil_images, transform = None):
        self.transform = transform
        self.data = []
        for j in range(len(list_of_pil_images)):
            
            image = list_of_pil_images[j][0]
            label = list_of_pil_images[j][1]
            self.data.append((image, label))  
                
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        image, label = self.data[index]
        if self.transform:
            sample = self.transform(image), torch.tensor(label)
        return sample


torch.manual_seed(0)

train_dataset, val_dataset = train_test_split(train_dataset, train_size = 0.8, test_size = 0.2, random_state = 0) #splitting

train_dataset = CustomDataset(train_dataset, transform = torchvision.transforms.ToTensor())
val_dataset = CustomDataset(val_dataset, transform = torchvision.transforms.ToTensor())
test_dataset = CustomDataset(test_dataset, transform = torchvision.transforms.ToTensor())


batch_size = 500
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_dataloader  = DataLoader(val_dataset,  batch_size=len(val_dataset), shuffle=False, num_workers=0)
test_dataloader  = DataLoader(test_dataset,  batch_size=len(test_dataset), shuffle=False, num_workers=0)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Training device: {device}")


# ### Definitions for the network


# I get the first tensor image and extract its pixel-size:
width = train_dataset.__getitem__(0)[0].shape[2]
height = train_dataset.__getitem__(0)[0].shape[1]

Ni = width * height
Nh1 = 16
Nh2 = 8
No = 10

layers = [Ni, Nh1, Nh2, No]


torch.manual_seed(0)
net = NonConvolutionalNet(layers)
#torch.save(net.state_dict(), 'net_parameters.torch') #save state before training
net.to(device)
net_settings = {
                'lr' : 1e-3,
                'epochs': 100,
                'optimizer': optim.SGD,
                'loss_fn' : nn.CrossEntropyLoss()
                }


# ### Training and test loop

net.train_and_validate(net_settings, train_dataloader, val_dataloader)

# ### Visualizing performances

# I am now defining these two functions so that I can use them again later with the CNN

def plot_performances(net):

    plt.figure(figsize = [12, 8])
    plt.plot(net.train_loss_log, label = 'Train loss')
    plt.plot(net.val_loss_log, label = 'Validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid()
    plt.legend()
    plt.show()

    plt.figure(figsize = [12, 8])
    plt.plot(np.divide(net.number_train_errors, train_dataset.__len__()), label = 'on training set')
    plt.plot(np.divide(net.number_val_errors, test_dataset.__len__()), label = 'on validation set'),
    plt.xlabel('Epoch')
    plt.ylabel('Percentage of missclassified images')
    plt.grid()
    plt.legend()
    plt.show()
    

def plot_classification(net):
    
    classes = {"0": "T-shirt/top",
               "1" : "Trouser",
               "2": "Pullover",
               "3": "Dress",
               "4": "Coat",
               "5": "Sandal",
               "6": "Shirt",
               "7": "Sneaker",
               "8": "Bag",
               "9": "Ankle Boot"
    }    

    images, labels = next(iter(test_dataloader))
    for index in range(10): # if > 20 I get a warning 
        img = images[index]
         # now I need to add an extra dimension to the tensor-image if I'm passing it to the convolutional layer
        if (type(net) is ConvolutionalNet):
            img = img.unsqueeze(1) 
        true_label = np.array2string(labels[index]) # I do this in order to print the right class and its name

        with torch.no_grad(): # So I speed up computations (also to avoid the error which tells me to detach())
            pb = net(img.to(device))
        pb = pb.data.cpu().numpy().squeeze()


        #plot side by side and using horizontal bars
        fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
        ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze(),cmap="gray") #cmap otherwise the image gets plotted in green for some reason
        ax1.axis('off')
        ax1.set_title(f'True class is {true_label}: {classes[true_label]}')
        ax2.barh(np.arange(10), pb)
        ax2.set_aspect(0.1)
        ax2.set_yticks(np.arange(10))
        ax2.set_yticklabels(np.arange(10))
        ax2.set_title('Class Probability found by classifier')
        ax2.set_xlim(0, 1.1)
        plt.tight_layout()
    

plot_performances(net)

plot_classification(net)


# Saving weights...


# First hidden layer
h1_w = net.fc1.weight.data.cpu().numpy()
h1_b = net.fc1.bias.data.cpu().numpy()

# Second hidden layer
h2_w = net.fc2.weight.data.cpu().numpy()
h2_b = net.fc2.bias.data.cpu().numpy()

# Output layer
out_w = net.out.weight.data.cpu().numpy()
out_b = net.out.bias.data.cpu().numpy()


### ...and plotting the histograms


# Weights histogram
fig, axs = plt.subplots(3, 1, figsize=(12,8))
axs[0].hist(h1_w.flatten(), 50)
axs[0].set_title('First hidden layer weights')
axs[1].hist(h2_w.flatten(), 50)
axs[1].set_title('Second hidden layer weights')
axs[2].hist(out_w.flatten(), 50)
axs[2].set_title('Output layer weights')
[ax.grid() for ax in axs]
plt.tight_layout()
plt.show()

### Cross Validation between standard implementations
#### Hyper-parameters



parameters = {
            'dropout' : [0, 0.5, 0.9],
            'epochs_num': [100],
            'learning_rates' : [1e-3,],
            'momenta': [None, 0, 0.5, 0.9],
            'optimizers': [ 'Adam', 'SGD'],
            'weight_decay' : [0, 0.1, 0.01],
            
}


from cv3 import ClassificationCV

layers = [784, 64, 32, 10]
cv = ClassificationCV(layers ,parameters)
cv.perform_CV(train_dataloader, val_dataloader)


# In[ ]:


plot_performances(cv.best_model.net)
plot_classification(cv.best_model.net)


# ## Part 2: Exploring other techniques : CNN


conv1 = 128
conv2 = 256
Ni = 7*7*conv2
Nh1 = 16
Nh2 = 8
No = 10

layers = [conv1, conv2, Ni, Nh1, Nh2, No]


torch.manual_seed(0)
cnn = ConvolutionalNet(layers)
cnn.to(device)
cnn_settings = {
                'lr' : 1e-3,
                'epochs': 100,
                'optimizer': optim.SGD,
                'loss_fn' : nn.CrossEntropyLoss()
                }



cnn.train_and_validate(cnn_settings, train_dataloader, val_dataloader)


# ### Visualizing weights...

# First conv layer
conv1_w = cnn.conv1.weight.data.cpu().numpy()
conv1_b = cnn.conv1.bias.data.cpu().numpy()

# Second conv layer
conv2_w = cnn.conv2.weight.data.cpu().numpy()
conv2_b = cnn.conv2.bias.data.cpu().numpy()

# First hidden layer
h1_w = cnn.fc1.weight.data.cpu().numpy()
h1_b = cnn.fc1.bias.data.cpu().numpy()

# Second hidden layer
h2_w = cnn.fc2.weight.data.cpu().numpy()
h2_b = cnn.fc2.bias.data.cpu().numpy()

# Output layer
out_w = cnn.out.weight.data.cpu().numpy()
out_b = cnn.out.bias.data.cpu().numpy()


# ### ...and plotting the histograms

# Weights histogram
fig, axs = plt.subplots(5, 1, figsize=(12,8))

axs[0].hist(conv1_w.flatten(), 50)
axs[0].set_title('First conv layer weights')
axs[1].hist(conv2_w.flatten(), 50)
axs[1].set_title('Second conv layer weights')
axs[2].hist(h1_w.flatten(), 50)
axs[2].set_title('First hidden layer weights')
axs[3].hist(h2_w.flatten(), 50)
axs[3].set_title('Second hidden layer weights')
axs[4].hist(out_w.flatten(), 50)
axs[4].set_title('Output layer weights')
[ax.grid() for ax in axs]
plt.tight_layout()
plt.show()


### Visualizing performances

plot_performances(cnn)
plot_classification(cnn)

### Cross validation on CNNs

#### Hyper-paramters


parameters = {
            'dropout' : [0, 0.5, 0.9],
            'epochs_num': [100],
            'learning_rates' : [1e-3,],
            'momenta': [None, 0, 0.5, 0.9],
            'optimizers': [ 'Adam', 'SGD'],
            'weight_decay' : [0, 0.1, 0.01],
            
}



layers = [conv1, conv2, Ni, Nh1, Nh2, No]

cv = ClassificationCV(layers ,parameters, convolutional = True)
cv.perform_CV(train_dataloader, val_dataloader)



