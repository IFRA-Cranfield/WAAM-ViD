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

# --- Constants ---
SEARCH_HEIGHT = 40
SEARCH_WIDTH = 50
ROI_HEIGHT = 190
MIN_ROI_WIDTH = 200
POOL_Y_OFFSET = 30
BLUR_KERNEL = 7
ROI_X_SHIFT_RATIO = 0.3
ROI_X_SHIFT_LEFT = 20
ROI_Y_SHIFT_UP = 90

def find_pool_roi_under_light(frame):
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    h_img, w_img = img_gray.shape  # Get image dimensions

    # Apply Gaussian blur if kernel is valid
    blurred = cv2.GaussianBlur(img_gray, (BLUR_KERNEL, BLUR_KERNEL), 0) if BLUR_KERNEL % 2 == 1 and BLUR_KERNEL > 0 else img_gray

    if blurred.size == 0:
        return None  # Abort if image is empty

    _, _, _, maxLoc_main = cv2.minMaxLoc(blurred)  # Brightest point
    main_light_x, main_light_y = maxLoc_main

    # Define search region around light
    y_start = min(h_img - 1, main_light_y)
    y_end = min(h_img, y_start + SEARCH_HEIGHT)
    x_start = max(0, main_light_x - SEARCH_WIDTH // 2)
    x_end = min(w_img, x_start + SEARCH_WIDTH)

    if y_start >= y_end or x_start >= x_end:
        return None  # Invalid region

    search_sub = blurred[y_start:y_end, x_start:x_end]  # Crop search region
    if search_sub.size == 0:
        return None

    _, _, _, maxLoc_s = cv2.minMaxLoc(search_sub)  # Brightest point in region
    target_x = maxLoc_s[0] + x_start
    target_y = maxLoc_s[1] + y_start

    # Define ROI size
    final_w = max(ROI_HEIGHT, MIN_ROI_WIDTH)
    final_h = ROI_HEIGHT

    # Compute ROI top-left corner
    roi_x = target_x - int(final_w * ROI_X_SHIFT_RATIO) - ROI_X_SHIFT_LEFT
    roi_y = target_y + POOL_Y_OFFSET - ROI_Y_SHIFT_UP

    # Clamp ROI to image bounds
    roi_x = max(0, min(roi_x, w_img - final_w))
    roi_y = max(0, min(roi_y, h_img - final_h))

    final_w = min(final_w, w_img - roi_x)
    final_h = min(final_h, h_img - roi_y)

    if final_w <= 0 or final_h <= 0:
        return None  # Invalid ROI

    return int(roi_x), int(roi_y), int(final_w), int(final_h)

    
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
