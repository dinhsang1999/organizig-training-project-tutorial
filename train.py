"""
Defines:
    loss function
    optimizer
    scheduler
    epoch loop
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split


##### CONSTANTS #####
NUM_CLASSES = 8
IMAGE_SIZE = 224                              # Image size (224x224)
DATA_DIR = 'data/retina/train'

#### HYPER PARAMETERS #####
BATCH_SIZE = 32                             
LEARNING_RATE = 0.0001
LEARNING_RATE_SCHEDULE_FACTOR = 0.1           # Parameter used for reducing learning rate
LEARNING_RATE_SCHEDULE_PATIENCE = 5           # Parameter used for reducing learning rate
MAX_EPOCHS = 100                              # Maximum number of training epochs
TRAINING_TIME_OUT=3600*10
NUM_WORKERS = 4


# Load model
from src.model import RetinaModel
model = RetinaModel(NUM_CLASSES)

# Loss function
criterion = nn.BCELoss()

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE,
                            betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-5)

# Learning rate reduced gradually during training
lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer, 
                        factor=LEARNING_RATE_SCHEDULE_FACTOR,
                        patience=LEARNING_RATE_SCHEDULE_PATIENCE,
                        mode='max', verbose=True)
                    

def epoch_training(epoch, model, train_dataloader, device, loss_criteria, optimizer, mb):
    pass

def train(device, model, train_dataloader, val_dataloader, max_epochs, loss_criteria, optimizer, lr_scheduler):
    pass



if __name__ == '__main__':
    ### Load data ###
    # Load label information from CSV train file  TODO: reconsider change `data` to `label`
    data = pd.read_csv('retina_dataset/train.csv')
    train_data, val_data = train_test_split(data, test_size=0.2, random_state=2020)

    # Create dataset and dataset dataloader
    data_dataset = RetinaDataSet(DATA_DIR + '/train', label, (IMAGE_SIZE, IMAGE_SIZE), true)
    data_dataloader = DataLoader(dataset=data_dataset, batch_size=BATCH_SIZE, 
                            shuffle=True, num_worker=NUM_WORKERS, pin_memory=True)

    # Create train dataset and train dataloader
    train_dataset = RetinaDataset(DATA_DIR + '/train', train_data, (IMAGE_SIZE,IMAGE_SIZE), True)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, 
                            shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)  

    # Create val dataset and val dataloader
    val_dataset = RetinaDataset(DATA_DIR + '/train', val_data, (IMAGE_SIZE,IMAGE_SIZE), True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, 
                            shuffle=None, num_workers=4, pin_memory=True)     

    ### Train ###
    device = "cuda" if torch.cuda.is_available else "cpu"
    train(device, model, train_dataloader, val_dataloader, MAX_EPOCHS,
            loss_criteria, optimizer, lr_scheduler)