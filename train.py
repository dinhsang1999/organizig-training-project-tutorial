"""
Defines:
    loss function
    optimizer
    scheduler
    epoch loop
"""
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from fastprogress import master_bar, progress_bar

from src.dataset import RetinaDataset
from src.model import RetinaModel


##### CONSTANTS #####
IMAGE_SIZE = 224                              # Image size (224x224)
DATA_DIR = '/Users/oddphoton/Projects/vietai/vietai_advance_w1b_retinal_disease_classificaton'

#### HYPER PARAMETERS #####
BATCH_SIZE = 32                             
LEARNING_RATE = 0.0001
LEARNING_RATE_SCHEDULE_FACTOR = 0.1           # Parameter used for reducing learning rate
LEARNING_RATE_SCHEDULE_PATIENCE = 5           # Parameter used for reducing learning rate
MAX_EPOCHS = 100                              # Maximum number of training epochs
TRAINING_TIME_OUT=3600*10
NUM_WORKERS = 0
                    

def epoch_training(epoch, model, train_dataloader, device, loss_criteria, optimizer, mb):
    model.train()
    training_loss = 0
    for batch, (images, labels,_) in enumerate(progress_bar(train_dataloader, parent=mb)): # TODO: make this loop a separate function
        # Move X, Y to device
        images = images.to(device)
        labels = labels.to(device)

        # Clear previous gradient
        optimizer.zero_grad()

        # Feed forward tge nidek
        pred = model(images)
        loss = loss_criteria(pred, labels)

        # Back probagation
        loss.backward()

        # Update parameters()
        optimizer.step()

        # Update training loss after each batch
        training_loss += loss.item()

        mb.child.comment = f'Training loss: {round(training_loss/(batch + 1), 5)}'

    del images, labels, loss

    return training_loss/len(train_dataloader)

def train(device, model, train_dataloader, val_dataloader, max_epochs, loss_criteria, optimizer, lr_scheduler):
    # initialisation
    best_score = 0
    training_losses = []

    # Config progress bar | For better presentation during training
    mb = master_bar(range(MAX_EPOCHS))
    mb.names = ['Training loss', 'Validation loss', 'Validation F1-score']

    # TODO: add time measurement

    # Training each epoch 
    for epoch in mb:
        mb.main_bar.comment = f'Best F1 score: {best_score}'
        
        # Training batches until covering all samples| Where model is updated ("learning")
        training_loss_epoch = epoch_training(epoch, model, train_dataloader, device, loss_criteria, optimizer, mb)

        training_losses.append(training_loss_epoch)

        mb.write('-Finis epoch {} | train loss: {:4.f}'.format(epoch, train_loss))

        # TODO: Add validation loss and f1 score
        # TODO: Add update learning rate (according to validation f1 score)
        # TODO: Add training chart
        # TODO: Add saving model


if __name__ == '__main__':
    ### Load data ###
    print('----- Loading data ... -----')
    # Load label information from CSV train file  TODO: reconsider change `data` to `label`
    labels = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    train_data, val_data = train_test_split(labels, test_size=0.2, random_state=2020)

    # Create dataset and dataset dataloader
    data_dataset = RetinaDataset(os.path.join(DATA_DIR, 'train'), labels, (IMAGE_SIZE, IMAGE_SIZE), True)
    data_dataloader = DataLoader(dataset=data_dataset, batch_size=BATCH_SIZE, 
                            shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)

    # Create train dataset and train dataloader
    train_dataset = RetinaDataset(os.path.join(DATA_DIR, 'train'), train_data, (IMAGE_SIZE,IMAGE_SIZE), True)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, 
                            shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)  

    # Create val dataset and val dataloader
    val_dataset = RetinaDataset(os.path.join(DATA_DIR, 'train'), val_data, (IMAGE_SIZE,IMAGE_SIZE), True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, 
                            shuffle=None, num_workers=4, pin_memory=True)     
    print('Done')

    ### Load model configuration ###
    print('----- Loading model configuration ... -----')
    # Load model
    NUM_CLASSES = labels.shape[1] - 1 # minus first column
    model = RetinaModel(NUM_CLASSES)

    # Loss function
    loss_criteria = nn.BCELoss()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE,
                                betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-5)

    # Learning rate reduced gradually during training
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                            optimizer, 
                            factor=LEARNING_RATE_SCHEDULE_FACTOR,
                            patience=LEARNING_RATE_SCHEDULE_PATIENCE,
                            mode='max', verbose=True)
    print('Done')
    
    ### Train ###
    print('----- Training model ... -----')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train(device, model, train_dataloader, val_dataloader, MAX_EPOCHS,
            loss_criteria, optimizer, lr_scheduler)
    print('Done')