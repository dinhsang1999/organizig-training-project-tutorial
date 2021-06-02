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
		pass

	def epoch_train(self, model, train_iterator, optimizer, loss_criteria, device): # Epoch training
		'''
		Training rountine in each epoch
		Returns:
			train_loss
		'''
		model.train()
		training_loss = 0
		for batch, (images, labels,_) in enumerate(train_iterator):
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

		del images, labels, loss

		return training_loss/len(train_iterator)

	def epoch_evaluate(self, model, val_iterator, loss_criteria, device): # Epoch evaluating
		'''
		Validating model after each epoch
		Returns:
			val_loss
			val_f1_score
		'''
		# Switch model to evaluation mode
		model.eval()

		val_loss = 0                                   # Total loss of model on validation set
		out_pred = torch.FloatTensor().to(device)      # Tensor stores prediction values
		out_gt = torch.FloatTensor().to(device)        # Tensor stores groundtruth values

		with torch.no_grad(): # Turn off gradient
			# For each batch
			for step, (images, labels,_) in enumerate(val_iterator):
				# Move images, labels to device (GPU)
				images = images.to(device)
				labels = labels.to(device)

				# Update groundtruth values
				out_gt = torch.cat((out_gt,  labels), 0)

				# Feed forward the model
				ps = model(images)
				loss = loss_criteria(ps, labels)

				# Update prediction values
				out_pred = torch.cat((out_pred, ps), 0)

				# Update validation loss after each batch
				val_loss += loss

		# Clear memory
		del images, labels, loss
		if torch.cuda.is_available(): 
			torch.cuda.empty_cache()

		# return validation loss, and metric score
		return val_loss/len(val_iterator), np.array(f1_score(out_gt, out_pred)).mean()

	def set_up_training_data(self):
		'''
		Return: 
			train_dataloader, val_dataloader
		'''
		print('----- Setting up data ... -----')
		# Load label information from CSV train file  TODO: reconsider change `data` to `label`
		labels = pd.read_csv(self.DATA_DIR_TRAIN_LABEL)
		train_data, val_data = train_test_split(labels, test_size=self.VAL_RATIO, random_state=2020)

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

	def set_up_training(self):
		'''
		Return:
			model, optimizer, criterion, scheduer, device
		'''
		### Load model configuration ###
		print('----- Loading model configuration ... -----')
		# Load model
		NUM_CLASSES = self.labels.shape[1] - 1 # minus first column
		model = RetinaModel(NUM_CLASSES)

		# Loss function
		loss_criteria = nn.BCELoss()

		# Optimizer
		optimizer = optim.Adam(model.parameters(), lr=self.LEARNING_RATE_START,
									betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-5)

		# Learning rate reduced gradually during training
		lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
								optimizer, 
								factor=self.LEARNING_RATE_SCHEDULE_FACTOR,
								patience=self.LEARNING_RATE_SCHEDULE_PATIENCE,
								mode='max', verbose=True)
		
		device = "cuda" if torch.cuda.is_available() else "cpu"

		print('Done')
		return model, optimizer, loss_criteria, lr_scheduler, device

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
		'''
		Convert epoch time (ms) to minutes and seconds
		'''
		elapsed_time = end_time - start_time
		elapsed_mins = int(elapsed_time / 60)
		elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
		return elapsed_mins, elapsed_secs

