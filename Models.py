import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torch import optim
import torch.nn.functional as F
import torch.nn as nn
import torch

###
# This module contains helper functions, and classes, for training PyTorch Neural Networks and was created
# by following this tutorial --> https://pytorch.org/tutorials/beginner/nn_tutorial.html#switch-to-cnn.
###

# Calculate the loss for a single training batch
def __loss_batch(model, loss_func, xb, yb, opt=None):
    
    # Make predictions and get loss
    loss = loss_func(model(xb), yb)

    # Backpropagate if an optimizer is specified
    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    # Return loss and length
    return loss.item(), len(xb)

# Fit an arbitrary model
def fit(epochs, model, loss_func, opt, train_dl, valid_dl):

    model = model.double()

    for epoch in range(epochs):
        
        model.train() # Put model in training mode
        for xb, yb in train_dl: # cycle through samples in training batch
            print(xb.dtype)
            __loss_batch(model, loss_func, xb, yb, opt)

        model.eval() # Put the model in evaluation mode
        with torch.no_grad(): # Don't update the gradients
            losses, nums = zip( # Calculate the total loss for the validation set
                *[__loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl]
            )
        
        # Calculate and print validation loss
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        print(f'Epoch {epoch}: Validation loss == {val_loss}')

# Returns DataLoader objects for fitting models, given training data, 
# validation data, and a training batch size
def get_dataloaders(train_ds, valid_ds, bs):
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(valid_ds, batch_size=bs * 2),
    )

# Lambda PyTorch Module that can be inserted into a "Sequential"
# object to perform arbitrary lambda operations on NN layers.
class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)