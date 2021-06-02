from src.trainer import BaselineClassifier
import torch
import json
import time
import os


def train():
	best_val_loss = float('inf') # Initial best validation loss

	# Load parameters
	params = json.load(open('./config/train_config.json', 'r'))
	print(params)

	# LOAD TRAINER
	trainer = BaselineClassifier(**params)
	# Set up DataLoader
	train_iterator, val_iterator = trainer.set_up_training_data()
	# Set up training params
	model, optimizer, loss_criteria, ls_scheduler, device = trainer.set_up_training()

	# TRAINING
	for epoch in range(trainer.MAX_EPOCHS):
		start_time = time.monotonic()
		print(f'Epoch {epoch+1}/{trainer.MAX_EPOCHS}')

		train_loss = trainer.epoch_train(model, train_iterator, optimizer, loss_criteria, device)
		val_loss, val_f1 = trainer.epoch_evaluate(model, val_iterator, loss_criteria, device)

		if val_loss < best_val_loss:
			best_val_loss = val_loss
			if not os.path.exists('./model'):
				os.makedirs('./model')
			torch.save(model.state_dict(), trainer.SAVE_MODEL_PATH)

		end_time = time.monotonic()

		epoch_mins, epoch_secs = trainer.epoch_time(start_time, end_time)

		print(f'Epoch: {epoch+1:02} | Epoch Time" {epoch_mins}m {epoch_secs}s')
		print(f'\t Train Loss: {train_loss:.3f}')
		print(f'\t Valid Loss: {val_loss:.3f} | Valid F1: {val_f1*100:6.2f}%')
		
	print('TRAINING DONE')

if __name__ == '__main__':
	train()
