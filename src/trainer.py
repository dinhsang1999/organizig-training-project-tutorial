import os
import time
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
from src.utils import f1_score

class Trainer(object):
	def __init__(self, **args):
		for key in args:
			setattr(self, key.upper(), args[key])

class BaselineClassifier(Trainer):
	def __init__(self, **args):
		super(BaselineClassifier, self).__init__(**args)
		# Load label information from CSV train file  TODO: reconsider change `data` to `label`
		self.labels = pd.read_csv(self.DATA_DIR_TRAIN_LABEL)
	
	def train(self, mode, iterator, optimizer, criterion, scheduler, device): # Epoch training
		'''
		Main training function
		'''

	def epoch_train(self, model, iterator, criterion, device, loss_criteria, optimizer, mb): # Epoch training
		pass

	def epoch_evaluate(self, model, iterator, criterion, device, loss_criteria, mb): # Epoch evaluating
		pass

	def set_up_training_data(self):
		'''
		Return: 
			train_dataloader, val_dataloader
		'''
		print('----- Setting up data ... -----')
		# Load label information from CSV train file  TODO: reconsider change `data` to `label`
		labels = pd.read_csv(self.DATA_DIR_TRAIN_LABEL)
		train_data, val_data = train_test_split(labels, test_size=self.VAL_RATIO, random_state=2020)

		# Create dataset and dataset dataloader
		data_dataset = RetinaDataset(self.DATA_DIR_TRAIN_IMAGES, labels, (self.IMAGE_SIZE, self.IMAGE_SIZE), True)
		data_dataloader = DataLoader(dataset=data_dataset, batch_size=self.BATCH_SIZE, 
								shuffle=True, num_workers=self.NUM_WORKERS, pin_memory=True)

		# Create train dataset and train dataloader
		train_dataset = RetinaDataset(self.DATA_DIR_TRAIN_IMAGES, train_data, (self.IMAGE_SIZE, self.IMAGE_SIZE), True)
		train_dataloader = DataLoader(dataset=train_dataset, batch_size=self.BATCH_SIZE, 
								shuffle=True, num_workers=self.NUM_WORKERS, pin_memory=True)  

		# Create val dataset and val dataloader
		val_dataset = RetinaDataset(self.DATA_DIR_TRAIN_IMAGES, val_data, (self.IMAGE_SIZE, self.IMAGE_SIZE), True)
		val_dataloader = DataLoader(dataset=val_dataset, batch_size=self.BATCH_SIZE, 
								shuffle=None, num_workers=self.NUM_WORKERS, pin_memory=True)     
		print('Done')
		return train_dataloader, val_dataloader

	def set_up_training(self, train_iterator):
		'''
		Return:
			model, optimizer, criterion, scheduer, device
		'''
		pass

	def train_test_split(self):
		'''
		Return:
			train_data, val_data
		'''
		pass

	def get_train_info(self, train_data):
		pass

	def get_model(self, **architecture):
		pass

	def epoch_time(self, start_time, end_time):
		pass
