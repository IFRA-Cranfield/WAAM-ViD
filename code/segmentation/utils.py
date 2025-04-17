import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2

""" Seeding randomness """
# This function was taken from PyTorch tutorial by idiot developer
def seeding(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

""" Calculate the time taken """
# This function was taken from PyTorch tutorial by idiot developer
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

""" Early Stop """
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

""" Training loop """
def training_loop (model, loader, optimizer, dice_loss,
                   device=torch.device('cuda')):
    epoch_loss = 0.0 
    
    model.train()
    for x,y in loader:
        # split data to image and mask
        image = x
        mask = y
        
        # send data to gpu for accelerated process
        image = image.to(device, dtype=torch.float32)
        mask = mask.to(device, dtype=torch.float32)
        
        # zero the gradients before backpropagation
        optimizer.zero_grad()
        
        # feed data to model and get predicted mask
        pred_mask = model(image)
        if isinstance(pred_mask, dict) and "out" in pred_mask:
            pred_mask = pred_mask["out"]

        # calculate loss value by comparing predicted and original mask
        # perform backpropagation, update parameters and calculate epoch loss
        loss = dice_loss(pred_mask, mask)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    epoch_loss = epoch_loss/len(loader) 
    
    return epoch_loss

""" Validation loop """
def validate_loop (model, loader, dice_loss, device=torch.device('cuda')):
    epoch_loss = 0.0
    
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            # split data to image and mask
            image = x
            mask = y
            
            # send data to gpu for accelerated process
            image = image.to(device, dtype=torch.float32)
            mask = mask.to(device, dtype=torch.float32)
            
            # feed data to model and get predicted mask
            pred_mask = model(image)
            if isinstance(pred_mask, dict) and "out" in pred_mask:
                pred_mask = pred_mask["out"]

            # calculate loss value by comparing predicted and original mask
            loss = dice_loss(pred_mask, mask)
            epoch_loss += loss.item()
    
    epoch_loss = epoch_loss/len(loader) 
    
    return epoch_loss
    
""" Plotting figures """
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
