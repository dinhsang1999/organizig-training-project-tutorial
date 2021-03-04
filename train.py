from src.trainer import BaselineClassifier
import torch
import json
import time

def train():
	best_valid_loss = float('inf') # Initial best validation loss

	# Load parameters
	params = json.load(open('./config/train_config.json', 'r'))
	print(params)

	# LOAD TRAINER
	trainer = BaselineClassifier(**params)
	# Set up DataLoader
	train_iterator, valid_iterator = trainer.set_up_training_data()
	# Set up training params
	model, optimizer, criterion, scheduler, device = trainer.set_up_training(train_iterator)

	# TRAINING
	for epoch in range(trainer.epochs):
		start_time = time.monotonic()

		train_loss, train_acc = trainer.train(model, train_iterator, optimizer, criterion, scheduler, device)
		valid_loss, valid_acc = trainer.evaluate(model, valid_iterator, criterion device)

		if valid_loss < best_valid_loss:
			best_valid_loss = valid_loss
			torch.save(model.state_dict(), trainer.save_model_path)

		end_time = time.monotonic()

		epoch_mins, epoch_secs = trainer.epoch_time(start_time, end_time)

		print(f'Epoch: {epoch+1:02} | Epoch Time" {epoch_mins}m {epoch_secs}s')
		print(f'\t Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:6.2f}%')
		print(f'\t Valid Loss: {valid_loss:.3f} | Valid Acc: {valid_acc*100:6.2f}%')
		
	print('TRAINING DONE')

if __name__ == '__main__':
	train()
