

class Trainer(object):
	def __init__(self, **args):
		for key in args:
			setattr(self, key, args[key])

class BaselineClassifier(Trainer):
	def __init__(self, **args):
		super(BaselineClassifier, self).__init__(**args)
	
	def train(self, mode, iterator, optimizer, criterion, scheduler, device): # Epoch training
		pass

	def evaluate(self, model, iterator, criterion, device): # Epoch evaluating
		pass

	def set_up_training_data(self):
		pass

	def set_up_training(self, train_iterator):
		pass

	def train_test_split(self):
		pass

	def get_train_info(self, train_data):
		pass

	def get_model(self, **architecture):
		pass

	def epoch_time(self, start_time, end_time):
		pass
