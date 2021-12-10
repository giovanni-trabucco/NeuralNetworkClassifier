from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from network_definitions import RegressionNet, NonConvolutionalNet, ConvolutionalNet
from colorama import Fore,Style #so the print below is easier to read
import itertools as it

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class RegressionCV():
    def __init__(self, layers: list, parameters: dict):
        self.results = []
        self.best_model = None
        self.parameters = parameters
        self.layers = layers
    
    def perform_CV(self, train_dataloader: DataLoader, val_dataloader: DataLoader): #Search procedure
        train_dataloader, val_dataloader = train_dataloader, val_dataloader
        parameters = self.parameters
        allNames = sorted(parameters)
        combinations = it.product(*(parameters[Name] for Name in allNames))
        list_combinations = list(combinations)
        
        for j, model in enumerate(list_combinations):
            dropout = model[0]
            epoch_n = model[1]
            l_rate = model[2]
            momentum = model[3]
            optimz = model[4]
            decay = model[5]
            
            net = RegressionNet(self.layers, dropout)
            net.to(device)
            loss_fn = nn.MSELoss()
            
            #First I do some checks...
            if (optimz is 'Adam' and momentum is None):
                optimizer = optim.Adam(params = net.parameters(), lr = l_rate, weight_decay = decay) # I can't pass 'momentum' to these
                print('######################')
                print(f'NOW TESTING: optim = {optimz}, lr = {l_rate}, num_epochs = {epoch_n}, weight_decay = {decay}, dropout = {dropout}\n')
            elif (optimz is 'SGD' and momentum is not None):
                optimizer = optim.SGD(params = net.parameters(), lr = l_rate, momentum = momentum, weight_decay = decay)
                print('######################')
                print(f'NOW TESTING: optim = {optimz}, lr = {l_rate}, num_epochs = {epoch_n}, momentum = {momentum}, weight_decay = {decay}, dropout = {dropout}\n')
            else:
                continue # So for example (SGD with momentum = None) or (Adam with momentum != None) get not excecuted
                
            #And than I start the real algotithm
            num_epochs = epoch_n
            train_loss_log = []
            val_loss_log = []

            for epoch_num in range (num_epochs):

                ## TRAIN
                train_loss = []
                net.train()
                for sample_batched in train_dataloader:
                    x_batch = sample_batched[0].to(device)
                    label_batch = sample_batched[1].to(device)

                    out = net(x_batch) #compute output ...
                    loss = loss_fn(out, label_batch) #... and loss function
                    net.zero_grad()
                    loss.backward() #gradient
                    optimizer.step() #update the weights

                    loss_batch = loss.detach().cpu().numpy()
                    train_loss.append(loss_batch) #train loss for this epoch

                #I have ended the epoch
                train_loss = np.mean(train_loss) 
                train_loss_log.append(train_loss) # Save average train loss

                ## VALIDATION
                val_loss = []
                net.eval()
                with torch.no_grad(): 
                    for sample_batched in val_dataloader:
                        # Move data to device
                        x_batch = sample_batched[0].to(device)
                        label_batch = sample_batched[1].to(device)

                        out = net(x_batch) #compute output ...
                        loss = loss_fn(out, label_batch) #... and loss function

                        loss_batch = loss.detach().cpu().numpy()
                        val_loss.append(loss_batch) #validation loss for this epoch

                    #I have ended the epoch
                    val_loss = np.mean(val_loss)
                    val_loss_log.append(val_loss) # Save average validation loss

            #... and now I store the info in the corresponding Result object...   
            result = self.Result(self.parameters, train_loss_log, val_loss_log, net)
            if (j == 0): #for the first iteration
                self.best_model = result

            curr_train_loss_log = result.mean_train_loss_log
            curr_val_loss_log = result.mean_val_loss_log

            #to compute the percentage of difference of the current from the best
            val_error_percentage_change = curr_val_loss_log / self.best_model.mean_val_loss_log - 1

            print(f'average train loss is: {curr_train_loss_log:.2f}')
            if (j == 0):
                print(f'average validation loss is: {curr_val_loss_log:.2f}') #first iter: no change wrt best model
            else:
                print(f'average validation loss is: {curr_val_loss_log:.2f} ({val_error_percentage_change*100:+.{2}f}% change wrt best model)')
                
            if (val_error_percentage_change <= 0):
                self.best_model = result #update the new best model
                print(Fore.GREEN + "THIS IS THE BEST MODEL UP UNTIL NOW")
            print(Style.RESET_ALL)

            self.results.append(result) #... and append the model to the list of models

############# THIS IS THE END OF THE SEARCH PROCEDURE #############   

        
#Here I am defining the inner class to store the information I need 
    class Result():
        
        def __init__(self, parameters: dict, train_loss_log, val_loss_log, net):
            self.net = net
            net.train_loss_log = train_loss_log # I first write these two in the network object, and then I keep them also here for convenience
            net.val_loss_log = val_loss_log
            
            self.train_loss_log = train_loss_log
            self.val_loss_log = val_loss_log
            self.mean_train_loss_log = np.mean(self.train_loss_log)
            self.mean_val_loss_log = np.mean(self.val_loss_log)    
            
            
            self.params = parameters

            # # ora voglio aggiungere il paramentro momentum nel caso in cui io stia usando SGD e non Adam
            # if (momentum is not None):
            #     self.params['momentum'] = momentum
                        
        
        def fit(self, data: torch.Tensor):
            return self.net(data)
        

##################################################
##################################################
##################################################
##################################################


class ClassificationCV():
    def __init__(self, layers: list, parameters: dict, convolutional = False):
        self.results = []
        self.best_model = None
        self.parameters = parameters
        self.layers = layers
        self.isConvolutional = convolutional
    
    def perform_CV(self, train_dataloader: DataLoader, val_dataloader: DataLoader): #Search procedure
        
        train_dataloader, val_dataloader = train_dataloader, val_dataloader
        parameters= self.parameters
        allNames = sorted(parameters) #otherwise I may risk to lose key-value pairs. However I build the dict so that it is already ordered
        combinations = it.product(*(parameters[Name] for Name in allNames))
        list_combinations = list(combinations)
        
        for j, model in enumerate(list_combinations):
            dropout = model[0]
            epoch_n = model[1]
            l_rate = model[2]
            momentum = model[3]
            optimz = model[4]
            decay = model[5]
            
            
            if (self.isConvolutional is False):
                net = NonConvolutionalNet(self.layers, dropout)
            
            else:
                net = ConvolutionalNet(self.layers, dropout)
            
            net.to(device)
            loss_fn = nn.CrossEntropyLoss()
            #First I do some checks...
            if (optimz is 'Adam' and momentum is None): # I can't pass 'momentum' to these
                optimizer = optim.Adam(params = net.parameters(), lr = l_rate)
                print('############################################\n')                                
                print(f'NOW TESTING: optim = {optimz}, lr = {l_rate}, num_epochs = {epoch_n}, weight_decay = {decay}, dropout = {dropout}\n')
            elif (optimz is 'SGD' and momentum is not None):
                optimizer = optim.SGD(params = net.parameters(), lr = l_rate, momentum = momentum)
                print('############################################\n')                                
                print(f'NOW TESTING: optim = {optimz}, lr = {l_rate}, num_epochs = {epoch_n}, momentum = {momentum}, weight_decay = {decay}, dropout = {dropout}\n')
            else:
                continue                
                
            #And than I start the algotithm
            num_epochs = epoch_n
            train_loss_log = []
            val_loss_log = []
            number_train_errors = []
            number_val_errors = []

            for epoch_num in range (num_epochs):

                ## TRAIN
                train_loss = []
                train_errors = []

                net.train()
                for sample_batched in train_dataloader:
                    x_batch = sample_batched[0].to(device)
                    label_batch = sample_batched[1].to(device)

                    out = net(x_batch) #compute output ...
                    loss = loss_fn(out, label_batch) #... and loss function
                    net.zero_grad()
                    loss.backward() #gradient
                    optimizer.step() #update the weights

                    loss_batch = loss.detach().cpu().numpy()
                    train_loss.append(loss_batch) #train loss for this epoch

                    '''now (with no_grad to save time) I find the number of misclassified images in each batch.
                    I do that by taking the (batch_size, 10)-tensor and by computing to which label gets
                    assigned the highest probability. I will then count how many times the (most probable)
                    predicted label is different from the true label'''

                    with torch.no_grad():
                        _, predictions = torch.max(out, dim = 1, keepdim = True)
                        wrong_indices = np.argwhere(predictions.squeeze(-1).cpu() != label_batch.cpu())
                        train_errors.append(wrong_indices.shape[1])

                #I have ended the epoch
                train_loss = np.mean(train_loss) 
                train_loss_log.append(train_loss) # Save average train loss

                total_errors_in_epoch = np.sum(train_errors)
                number_train_errors.append(total_errors_in_epoch)

                ## VALIDATION
                val_loss = []
                val_errors = []

                net.eval()
                with torch.no_grad(): 
                    for sample_batched in val_dataloader:
                        # Move data to device
                        x_batch = sample_batched[0].to(device)
                        label_batch = sample_batched[1].to(device)
                        
                        out = net(x_batch) #compute output ...
                        loss = loss_fn(out, label_batch) #... and loss function

                        loss_batch = loss.detach().cpu().numpy()
                        val_loss.append(loss_batch) #validation loss for this epoch

                        _, predictions = torch.max(out, dim = 1, keepdim = True)
                        val_errors.append((np.argwhere(predictions.squeeze(-1).cpu() != label_batch.cpu())).shape[1])                                          

                    #I have ended the epoch
                    val_loss = np.mean(val_loss)
                    val_loss_log.append(val_loss) # Save average validation loss

                    average_errors_in_epoch = int(np.mean(val_errors))
                    number_val_errors.append(average_errors_in_epoch)


            #... and now I store the info in the corresponding Result object...   
            result = self.Result(self.parameters, train_loss_log, val_loss_log,\
                                 number_train_errors, number_val_errors, net)
            #when doing cross-validation here I need to store the number of errors averaged on all epochs

            if (j == 0): #for the first iteration
                self.best_model = result

            curr_train_loss_log = result.mean_train_loss_log
            curr_val_loss_log = result.mean_val_loss_log

            #to compute the percentage of difference of the current from the best
            val_error_percentage_change = curr_val_loss_log / self.best_model.mean_val_loss_log - 1


            print(f'Average train loss is: {curr_train_loss_log:.2f}')
            print(f'Average number of misclassified TRAINING images: '\
                f'{int(np.mean(number_train_errors)/(train_dataloader.dataset.__len__()))}'\
                f' (that\'s about {np.mean(number_train_errors)/train_dataloader.dataset.__len__()* 100:.1f}%)\n')  
            
            if (j == 0):
                print(f'Average validation loss is: {curr_val_loss_log:.2f}') #first iter: no change wrt best model
            else:
                print(f'Average validation loss is: {curr_val_loss_log:.2f} ({val_error_percentage_change*100:+.{2}f}% change wrt best model)')
            print(f'Average number of misclassified VALIDATION images: '\
                f'{int(np.mean(number_val_errors)/(val_dataloader.dataset.__len__()))}'\
                f' (that\'s about {np.mean(number_val_errors) / val_dataloader.dataset.__len__()* 100:.1f}%)\n') 

            if (val_error_percentage_change <= 0):
                self.best_model = result #update the new best model
                print(Fore.GREEN + "THIS IS THE BEST MODEL UP UNTIL NOW")
                print(Style.RESET_ALL)

            self.results.append(result) #... and append the model to the list of models

############# THIS IS THE END OF THE SEARCH PROCEDURE #############   

        
#Here I am defining the inner class to store the information I need 
    class Result():
        
        def __init__(self, parameters: dict, train_loss_log, val_loss_log, number_train_errors, number_val_errors, net):
            
            self.net = net
            net.train_loss_log = train_loss_log  #I write these values in the network object for consistency...
            net.val_loss_log = val_loss_log
            net.number_train_errors = number_train_errors
            net.number_val_errors = number_val_errors
            
            self.train_loss_log = train_loss_log #...but I also keep them here for convenience
            self.mean_train_loss_log = np.mean(train_loss_log)
            self.val_loss_log = val_loss_log
            self.mean_val_loss_log = np.mean(val_loss_log)
            self.number_train_errors = number_train_errors
            self.number_val_errors = number_val_errors
            self.net = net
            
            self.params = parameters

            
            # # ora voglio aggiungere il paramentro momentum nel caso in cui io stia usando SGD e non Adam
            # if (self.parameters["momentum"] is not None):
            #     self.parameters['momentum'] = momentum
                        
        
        

        
##################################################
##################################################
##################################################
##################################################


class ConvolutionalCV():
    def __init__(self, layers: list, parameters: dict):
        self.results = []
        self.best_model = None
        self.parameters = parameters
        self.layers = layers
    
    def perform_CV(self, train_dataloader: DataLoader, val_dataloader: DataLoader): #Search procedure
        
        train_dataloader, val_dataloader = train_dataloader, val_dataloader
        parameters= self.parameters
        allNames = sorted(parameters) #otherwise I may risk to lose key-value pairs. However I build the dict so that it is already ordered
        combinations = it.product(*(parameters[Name] for Name in allNames))
        list_combinations = list(combinations)
        
        for j, model in enumerate(list_combinations):
            dropout = model[0]
            epoch_n = model[1]
            l_rate = model[2]
            momentum = model[3]
            optimz = model[4]
            decay = model[5]

            net = ConvolutionalNet(self.layers, dropout)
            #net.load_state_dict(torch.load('net_parameters.torch'))
                
            net.to(device)
            loss_fn = nn.CrossEntropyLoss()
            #First I do some checks...
            if (optimz is 'Adam' and momentum is None): # I can't pass 'momentum' to these
                optimizer = optim.Adam(params = net.parameters(), lr = l_rate)
                print('############################################\n')                                
                print(f'NOW TESTING: optim = {optimz}, lr = {l_rate}, num_epochs = {epoch_n}, weight_decay = {decay}, dropout = {dropout}\n')
            elif (optimz is 'SGD' and momentum is not None):
                optimizer = optim.SGD(params = net.parameters(), lr = l_rate, momentum = momentum)
                print('############################################\n')                                
                print(f'NOW TESTING: optim = {optimz}, lr = {l_rate}, num_epochs = {epoch_n}, momentum = {momentum}, weight_decay = {decay}, dropout = {dropout}\n')
            else:
                continue                
                
            #And than I start the algotithm
            num_epochs = epoch_n
            train_loss_log = []
            val_loss_log = []
            number_train_errors = []
            number_val_errors = []

            for epoch_num in range (num_epochs):

                ## TRAIN
                train_loss = []
                train_errors = []

                net.train()
                for sample_batched in train_dataloader:
                    x_batch = sample_batched[0].to(device)
                    label_batch = sample_batched[1].to(device)

                    out = net(x_batch) #compute output ...
                    loss = loss_fn(out, label_batch) #... and loss function
                    net.zero_grad()
                    loss.backward() #gradient
                    optimizer.step() #update the weights

                    loss_batch = loss.detach().cpu().numpy()
                    train_loss.append(loss_batch) #train loss for this epoch

                    '''now (with no_grad to save time) I find the number of misclassified images in each batch.
                    I do that by taking the (batch_size, 10)-tensor and by computing to which label gets
                    assigned the highest probability. I will then count how many times the (most probable)
                    predicted label is different from the true label'''

                    with torch.no_grad():
                        _, predictions = torch.max(out, dim = 1, keepdim = True)
                        wrong_indices = np.argwhere(predictions.squeeze(-1).cpu() != label_batch.cpu())
                        train_errors.append(wrong_indices.shape[1])

                #I have ended the epoch
                train_loss = np.mean(train_loss) 
                train_loss_log.append(train_loss) # Save average train loss

                total_errors_in_epoch = np.sum(train_errors)
                number_train_errors.append(total_errors_in_epoch)

                ## VALIDATION
                val_loss = []
                val_errors = []

                net.eval()
                with torch.no_grad(): 
                    for sample_batched in val_dataloader:
                        # Move data to device
                        x_batch = sample_batched[0]
                        x_batch = x_batch.view(x_batch.shape[0], -1).to(device)

                        label_batch = sample_batched[1].to(device)
                        out = net(x_batch) #compute output ...
                        loss = loss_fn(out, label_batch) #... and loss function

                        loss_batch = loss.detach().cpu().numpy()
                        val_loss.append(loss_batch) #validation loss for this epoch

                        _, predictions = torch.max(out, dim = 1, keepdim = True)
                        val_errors.append((np.argwhere(predictions.squeeze(-1).cpu() != label_batch.cpu())).shape[1])                                          

                    #I have ended the epoch
                    val_loss = np.mean(val_loss)
                    val_loss_log.append(val_loss) # Save average validation loss

                    average_errors_in_epoch = int(np.mean(val_errors))
                    number_val_errors.append(average_errors_in_epoch)


            #... and now I store the info in the corresponding Result object...   
            result = self.Result(self.parameters, train_loss_log, val_loss_log,\
                                 number_train_errors, number_val_errors, net)
            #when doing cross-validation here I need to store the number of errors averaged on all epochs

            if (j == 0): #for the first iteration
                self.best_model = result

            curr_train_loss_log = result.mean_train_loss_log
            curr_val_loss_log = result.mean_val_loss_log

            #to compute the percentage of difference of the current from the best
            val_error_percentage_change = curr_val_loss_log / self.best_model.mean_val_loss_log - 1


            print(f'Average train loss is: {curr_train_loss_log:.2f}')
            print(f'Average number of misclassified TRAINING images: '\
                f'{int(np.mean(number_train_errors)/(train_dataloader.dataset.__len__()))}'\
                f' (that\'s about {np.mean(number_train_errors)/train_dataloader.dataset.__len__()* 100:.1f}%)\n')  
            
            if (j == 0):
                print(f'Average validation loss is: {curr_val_loss_log:.2f}') #first iter: no change wrt best model
            else:
                print(f'Average validation loss is: {curr_val_loss_log:.2f} ({val_error_percentage_change*100:+.{2}f}% change wrt best model)')
            print(f'Average number of misclassified VALIDATION images: '\
                f'{int(np.mean(number_val_errors)/(val_dataloader.dataset.__len__()))}'\
                f' (that\'s about {np.mean(number_val_errors) / val_dataloader.dataset.__len__()* 100:.1f}%)\n') 

            if (val_error_percentage_change <= 0):
                self.best_model = result #update the new best model
                print(Fore.GREEN + "THIS IS THE BEST MODEL UP UNTIL NOW")
                print(Style.RESET_ALL)

            self.results.append(result) #... and append the model to the list of models

############# THIS IS THE END OF THE SEARCH PROCEDURE #############   

        
#Here I am defining the inner class to store the information I need 
    class Result():
        
        def __init__(self, parameters: dict, train_loss_log, val_loss_log, number_train_errors, number_val_errors, net):
            
            self.net = net
            net.train_loss_log = train_loss_log  #I write these values in the network object for consistency...
            net.val_loss_log = val_loss_log
            net.number_train_errors = number_train_errors
            net.number_val_errors = number_val_errors
            
            self.train_loss_log = train_loss_log #...but I also keep them here for convenience
            self.mean_train_loss_log = np.mean(train_loss_log)
            self.val_loss_log = val_loss_log
            self.mean_val_loss_log = np.mean(val_loss_log)
            self.number_train_errors = number_train_errors
            self.number_val_errors = number_val_errors
            self.net = net
            
            self.params = parameters

            
            # # ora voglio aggiungere il paramentro momentum nel caso in cui io stia usando SGD e non Adam
            # if (self.parameters["momentum"] is not None):
            #     self.parameters['momentum'] = momentum
                        
        
        
