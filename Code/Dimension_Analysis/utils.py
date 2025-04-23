import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

# seeding randomness
# This function was taken from PyTorch tutorial by idiot developer
def seeding(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

# calculate the time taken
# This function was taken from PyTorch tutorial by idiot developer
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

# early stop
# The original EarlyStopper class was coded by isle-of-gods 
#(https://stackoverflow.com/users/3807097/isle-of-gods)
# This is an adaptation of the original code
class EarlyStopper:
    def __init__(self, patience=1):
        self.patience = patience 
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        # This is ensures that the model does not overfit
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0 # resets counter
            return False
            
        # When valLoss >= min_valLoss, start counter  
        else: 
            self.counter += 1
            if self.counter >= self.patience:
                return True

# training loop
def training_loop (model, loader, optimizer, criterion, device=torch.device('cuda')):
    epoch_loss = 0.0 
    
    model.train()
    for batch in loader:
        # extract inputs and labels from the batch
        input1 = batch["predict_mask"].to(device, dtype=torch.float32)
        input1 = input1.repeat(1, 3, 1, 1)  # repeat input1 to match the expected input shape of the model

        input2 = torch.cat([
            batch["camera_matrix"].to(device, dtype=torch.float32),
            batch["distortion_coefficients"].to(device, dtype=torch.float32),
            batch["rotation_matrix"].to(device, dtype=torch.float32),
            batch["translation_vector"].to(device, dtype=torch.float32)
        ], dim=1)  # concatenate input2 features
        label = batch["width"].to(device, dtype=torch.float32).unsqueeze(1).squeeze(-1)  # width as label
        
        # zero the gradients before backpropagation
        optimizer.zero_grad()
        
        # forward pass
        outputs = model(input1, input2)

        # calculate loss value by comparing predicted and original mask
        # perform backpropagation, update parameters and calculate epoch loss
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    epoch_loss = epoch_loss/len(loader) 
    
    return epoch_loss

# validataion loop
def validate_loop (model, loader, criterion, device=torch.device('cuda')):
    epoch_loss = 0.0
    
    model.eval()
    with torch.no_grad():
        for batch in loader:
            # extract inputs and labels from the batch
            input1 = batch["predict_mask"].to(device, dtype=torch.float32)
            input1 = input1.repeat(1, 3, 1, 1)  # repeat input1 to match the expected input shape of the model

            input2 = torch.cat([
                batch["camera_matrix"].to(device, dtype=torch.float32),
                batch["distortion_coefficients"].to(device, dtype=torch.float32),
                batch["rotation_matrix"].to(device, dtype=torch.float32),
                batch["translation_vector"].to(device, dtype=torch.float32)
            ], dim=1)  # concatenate input2 features
            label = batch["width"].to(device, dtype=torch.float32).unsqueeze(1).squeeze(-1)  # width as label
            
            # forward pass
            outputs = model(input1, input2)

            # calculate loss value by comparing predicted and original mask
            loss = criterion(outputs, label)
            epoch_loss += loss.item()
    
    epoch_loss = epoch_loss/len(loader) 
    
    return epoch_loss

# testing loop
def testing_loop(model, loader, device=torch.device('cuda')):
    rmse_list = []
    mae_list = []
    r2_list = []

    with torch.no_grad():
        for batch in loader:
            # extract inputs and labels from the batch
            input1 = batch["predict_mask"].to(device, dtype=torch.float32)
            input1 = input1.repeat(1, 3, 1, 1)  # repeat input1 to match the expected input shape of the model

            input2 = torch.cat([
                batch["camera_matrix"].to(device, dtype=torch.float32),
                batch["distortion_coefficients"].to(device, dtype=torch.float32),
                batch["rotation_matrix"].to(device, dtype=torch.float32),
                batch["translation_vector"].to(device, dtype=torch.float32)
            ], dim=1)  # concatenate input2 features
            label = batch["width"].to(device, dtype=torch.float32).unsqueeze(1).squeeze(-1)  # width as label

            # forward pass
            outputs = model(input1, input2)

            # move outputs and labels to CPU
            outputs = outputs.cpu()
            label = label.cpu()

            # compute metrics
            rmse = torch.sqrt(torch.mean((outputs - label) ** 2)).item()  # Root Mean Squared Error
            mae = torch.mean(torch.abs(outputs - label)).item()  # Mean Absolute Error
            r2 = 1 - torch.sum((outputs - label) ** 2) / torch.sum((label - torch.mean(label)) ** 2).item()

            rmse_list.append(rmse)
            mae_list.append(mae)
            r2_list.append(r2)

    return rmse_list, mae_list, r2_list

# plotting functions
def plot_loss(train_loss, val_loss):
    plt.figure(dpi=700)
    plt.plot(train_loss, label = 'Training Loss')
    plt.plot(val_loss, label = 'Validation Loss')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()
    plt.savefig('Loss graph.png')

def plot_lr(lr):
    plt.figure(dpi=700)
    plt.plot(lr)
    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')
    plt.show()
    plt.savefig('Learning rate.png')
