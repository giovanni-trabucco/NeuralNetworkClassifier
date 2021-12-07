import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
torch.manual_seed(0)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class RegressionNet(nn.Module):
    
    def __init__(self, layers: list, dropout = 0):

        super().__init__()
        
        Ni = layers[0]
        Nh1 = layers[1]
        Nh2 = layers[2]
        No = layers[3]
        dropout = dropout
        
        self.fc1 = nn.Linear(in_features = Ni, out_features = Nh1)
        self.fc2 = nn.Linear(in_features = Nh1, out_features = Nh2)
        self.out = nn.Linear(in_features = Nh2, out_features = No)
        
        self.dropout = nn.Dropout (p = dropout)
        self.act = nn.ReLU()
        
        self.train_loss_log = []
        self.val_loss_log = []
        
        
    def forward(self, data, additional_out=False):
        x = data
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.out(x)

        return x

    
    def train_and_validate(self, net_settings: dict, train_dataloader: DataLoader, val_dataloader: DataLoader):
        #self.load_state_dict(torch.load('net_parameters.torch'))
        loss_fn = net_settings["loss_fn"]
        lr = net_settings["lr"]
        optimizer = net_settings["optimizer"](self.parameters(), lr = lr)
        num_epochs = net_settings["epochs"]
        
        train_dataloader = train_dataloader
        val_dataloader = val_dataloader
        

        for epoch_num in range(num_epochs):
            print(f'Epoch number: {epoch_num}')

            ### TRAIN
            train_loss = []
            self.train() 
            for sample_batched in train_dataloader:
                
                x_batch = sample_batched[0].to(device)
                label_batch = sample_batched[1].to(device)
                
                out = self(x_batch)
                loss = loss_fn(out, label_batch)                
                self.zero_grad()
                loss.backward()
                optimizer.step()

                loss_batch = loss.detach().cpu().numpy()
                train_loss.append(loss_batch)
                
            train_loss = np.mean(train_loss)
            print(f"Average train loss: {train_loss:.2f}")
            self.train_loss_log.append(train_loss)

            
            # VAL
            val_loss= []
            self.eval() 
            with torch.no_grad(): 
                for sample_batched in val_dataloader:
                    # Move data to device
                    x_batch = sample_batched[0].to(device)
                    label_batch = sample_batched[1].to(device)

                    out = self(x_batch)
                    loss = loss_fn(out, label_batch)
                   
                    loss_batch = loss.detach().cpu().numpy()
                    val_loss.append(loss_batch)

                # Save average validation loss
                val_loss = np.mean(val_loss)
                print(f"Average validation loss: {val_loss:.2f}\n")
                self.val_loss_log.append(val_loss)


##########################################################
##########################################################
##########################################################
##########################################################

class ClassificationNetworks(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        '''If I define the next four variables in the train_and_test() function and then I return them, the problem is that
        if I interrupt the execution of the function (maybe I don't want to wait until the end of all the epochs), they will not be returned.
        For this reason I will define them as object attributes.'''
           
        self.train_loss_log = []
        self.val_loss_log = []

        self.number_train_errors = []
        self.number_val_errors = []
    
    def train_and_validate(self, net_settings: dict, train_dataloader: DataLoader, val_dataloader: DataLoader):
            
        loss_fn = net_settings["loss_fn"]
        lr = net_settings["lr"]
        optimizer = net_settings["optimizer"](self.parameters(), lr = lr)
        num_epochs = net_settings["epochs"]
        
        train_dataloader = train_dataloader
        val_dataloader = val_dataloader
        
    
        for epoch in range(num_epochs):

            #TRAIN
            train_loss = []
            train_errors = []
            
            self.train()
            print(f'epoch number: {epoch}')
            for sample_batched in train_dataloader:
                x_batch = sample_batched[0].to(device)
                label_batch = sample_batched[1].to(device)

                out = self(x_batch)
                loss = loss_fn(out, label_batch)
                self.zero_grad()
                loss.backward()
                optimizer.step()


                loss_batch = loss.detach().cpu().numpy()
                train_loss.append(loss_batch)

                '''now (with no_grad to save time) I find the number of misclassified images in each batch.
                  I do that by taking the (batch_size, 10)-tensor and by computing to which label gets
                  assigned the highest probability. I will then count how many times the (most probable)
                  predicted label is different from the true label'''

                with torch.no_grad():
                    _, predictions = torch.max(out, dim = 1, keepdim = True) #returns the tensor with the predictions (most probable class)
                    
                    wrong_indices = np.argwhere(predictions.squeeze(-1).cpu() != label_batch.cpu()) # tensor containing the indexes of misclassifications...
                    train_errors.append(wrong_indices.shape[1]) #...but I'm interested in HOW MANY indices were different
                    # also, I needed to squeeze because prediction indices was of size (sample_batch, 1)...
                    #...while label_batch was (label_batch(=sample_batch), 0)

                    
            # end of first epoch, need to compute the mean over the batch_losses...
            print(f'Average training loss: {np.mean(train_loss):.2f}')
            self.train_loss_log.append(np.mean(train_loss))
            
            #..while here I need to sum all the errors computed in each batch and then average 
            total_errors_in_epoch = np.sum(train_errors)
            self.number_train_errors.append(total_errors_in_epoch)
            print(f'Average number of misclassified TRAINING images: '\
                  f'{total_errors_in_epoch}/{train_dataloader.dataset.__len__()}'\
                  f' (that\'s about {total_errors_in_epoch / train_dataloader.dataset.__len__()* 100:.1f}%)')


            #VAL
            val_loss = []
            val_errors = []
            
            self.eval()
            with torch.no_grad():
                for sample_batched in val_dataloader:
                    x_batch = sample_batched[0].to(device)
                    label_batch = sample_batched[1].to(device)

                    out = self(x_batch)
                    loss = loss_fn(out,label_batch)

                    loss_batch = loss.detach().cpu().numpy()
                    val_loss.append(loss_batch)


                    _, predictions = torch.max(out, dim = 1, keepdim = True)
                    val_errors.append((np.argwhere(predictions.squeeze(-1).cpu() != label_batch.cpu())).shape[1])    


                #end of first epoch, need to compute the mean over the batch_losses
                print(f'Average validation_loss: {np.mean(val_loss):.2f}')
                self.val_loss_log.append(np.mean(val_loss))
                
                average_errors_in_epoch = int(np.mean(val_errors))
                self.number_val_errors.append(average_errors_in_epoch)
                print(f'Average number of misclassified VALIDATION images: '\
                    f'{average_errors_in_epoch}/{val_dataloader.dataset.__len__()}'\
                      f' (that\'s about {average_errors_in_epoch / val_dataloader.dataset.__len__()* 100:.1f}%)\n')


        
    
class NonConvolutionalNet(ClassificationNetworks):        
    def __init__(self, layers: list, dropout = 0):    
        super().__init__()
        Ni = layers[0]
        Nh1 = layers[1]
        Nh2 = layers[2]
        No = layers[3]
        dropout = dropout
        
        self.fc1 = nn.Linear(in_features = Ni, out_features = Nh1)
        self.fc2 = nn.Linear(in_features = Nh1, out_features = Nh2)
        self.out = nn.Linear(in_features = Nh2, out_features = No)
        
        self.dropout = nn.Dropout (p = dropout)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim = 1)
            
    def forward(self, data):
        x = data
        x = x.view(x.shape[0], -1) #compressing the 28x28 pixels
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.out(x)
        x = self.softmax(x)
         
        return x


    
    
# class ConvolutionalNet(ClassificationNetworks):   
#     def __init__(self):
#         super().__init__()

#         self.cnn_layers = nn.Sequential(
#             # Defining a 2D convolution layer
#             nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(4),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             # Defining another 2D convolution layer
#             nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(4),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#         )

#         self.linear_layers = nn.Sequential(
#             nn.Linear(4 * 7 * 7, 10)
#         )

#     # Defining the forward pass    
#     def forward(self, x):
#         x = self.cnn_layers(x)
#         x = x.view(x.size(0), -1)
#         x = self.linear_layers(x)
#         return x    
        

    

    
    
class ConvolutionalNet(ClassificationNetworks):
        
    def __init__(self, layers: list, dropout = 0):
        super().__init__()
        conv1 = layers[0]
        conv2 = layers[1]
        Ni = layers[2]
        Nh1= layers[3]
        Nh2 = layers[4]
        No = layers[5]
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=conv1, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=conv1, out_channels=conv2, kernel_size=3, stride=1, padding=1)
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(in_features = Ni, out_features = Nh1)
        self.fc2 = nn.Linear(in_features = Nh1, out_features = Nh2)
        self.out = nn.Linear(in_features = Nh2, out_features = No)
        
        self.dropout = nn.Dropout(p = dropout)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim = 1)


    def forward(self, data):
        x = data
        x = self.conv1(x)
        x = self.act(x)
        x = self.max_pool2d(x)
        x = self.conv2(x)
        x = self.act(x)
        x = self.max_pool2d(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.out(x)
        x = self.softmax(x)

        return x
