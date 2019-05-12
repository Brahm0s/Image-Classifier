import argparse
import torch
from collections import OrderedDict
from os.path import isdir
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
import os

def arg_parser():
    # Define parser
	parser = argparse.ArgumentParser(description = "Inputs for Neural Networks Attributes")

	# Enter the directory for datasets
	parser.add_argument('--data_dir', type = str, help = 'Directory where data is present')
    
    # Add the architecture for model
	parser.add_argument('--arch', type = str, help = 'Choose architecture from torchvision.models as str')
    
    # Add filename in which to save the model
	parser.add_argument('--save_dir', type = str, help = 'Define save directory for checkpoints as str. If not specified then model will be lost.')
    
    # Add learning rate for the model
	parser.add_argument('--learning_rate', type = float, help = 'Define gradient descent learning rate as float')

    # Add the no of hidden units in the model
	parser.add_argument('--hidden_units', type = int, help = 'Hidden units for DNN classifier as int')

    # The total no of ephocs to be completed
	parser.add_argument('--epochs', type = int, help = 'Number of epochs for training as int')
    
    # Input if gpu or cpu should be used for prediction
    parser.add_argument('--processing', type = str, help = 'Either CPU or GPU')

    # Add all the arguments to args variable
	args = parser.parse_args()
	return args

# Function to train the network
def train_network(model, trainloader, validloader, device, criterion, optimizer, epochs = 5):
	# The no of times after which the model details will be printed
	print_every = 30
	steps = 0


	print("Training process initializing..\n")

# Starting the training process
	for e in range(epochs):
		running_loss = 0
		# Set the neural network to training mode        
		model.train()

		for ii, (inputs, labels) in enumerate(trainloader):
			steps+=1

			inputs, labels = inputs.to(device), labels.to(device)

			optimizer.zero_grad()

			# Calculating the output from the neural network
			output = model.forward(inputs)
			# Calculating the loss            
			loss = criterion(output,labels)
			# Starting the back propagation process          
			loss.backward()
			# Starting the optimizing the process            
			optimizer.step()

			running_loss += loss.item()

			if steps % print_every == 0:
				model.eval()
				# Calculating the accuracy of the model with current architecture
				with torch.no_grad():
                    
					valid_loss, accuracy = validation(model, validloader, criterion, device)

				# Printing the atributes
				print("epoch: {}/{} | ".format(e+1, epochs),
					"Training Loss: {:.4f} |".format(running_loss/print_every),
					"Validation Loss: {:.4f} | ".format(valid_loss/len(validloader)),
					"Validation Accuracy: {:.4f}".format(accuracy/len(validloader)))

				running_loss = 0
				model.train()

	return model

# Definitation of the validation model
def validation(model, testloader, criterion, device):
	# Calculating the model accuracy with current architecture    
	test_loss = 0
	accuracy = 0
    
	for ii, (inputs, labels) in enumerate(testloader):
        
		inputs, labels = inputs.to(device), labels.to(device)
        
		output = model.forward(inputs)
		test_loss += criterion(output, labels).item()
        
		ps = torch.exp(output)
		equality = (labels.data == ps.max(dim = 1)[1])
		accuracy += equality.type(torch.FloatTensor).mean()
	return test_loss, accuracy

# Calculating the final model acuraccy
def validate_model(model, testloader, device):

	correct = 0
	total = 0
	with torch.no_grad():
		model.eval()
		for data in testloader:
			images, labels = data
			images, labels = images.to(device), labels.to(device)
			outputs = model(images)
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()

	# Print the accuracy of the model    
	print('Accuracy achieved by the network on test images is: %d%%' %(100 * correct/ total))

# Function to save the model so that the trained neural network is not lost
def save_model(model, save_dir, train_data):

	if type(save_dir) == type(None):
		print("MOdel save directory not specified")
	else:
		if isdir(save_dir):
			path = os.path.join(save_dir, 'checkpoint.pth')

			model.class_to_idx = train_data.class_to_idx

			checkpoint = {'architecture': model.name,
						'classifer': model.classifier,
						'class_to_idx': model.class_to_idx,
						'state_dict': model.state_dict()}

			torch.save(checkpoint, path)
			print("moel saved")
		else:
			print("Directory not found, model will not be saved")


# Definition of the main function
def main():
	# Get all the entered arguments
	args = arg_parser()

	# Set the file directories
	data_dir = args.data_dir
	os.path.join(data_dir, 'train')
	os.path.join(data_dir, 'valid')
	os.path.join(data_dir, 'test')
    
	# Defining the transforms
	train_transforms = transforms.Compose([transforms.RandomRotation(30),
											transforms.RandomResizedCrop(224),
											transforms.RandomHorizontalFlip(),
											transforms.ToTensor(),
											transforms.Normalize([0.485, 0.456, 0.406],
											[0.229, 0.224, 0.225])])

	test_transforms = transforms.Compose([transforms.Resize(256),
											transforms.CenterCrop(224),
											transforms.ToTensor(),
											transforms.Normalize([0.485, 0.456, 0.406],
																[0.229, 0.224, 0.225])])

	# Loading the data and applying transforms
	train_data = datasets.ImageFolder(train_dir, transform = train_transforms)

	test_data = datasets.ImageFolder(test_dir, transform = test_transforms)

	valid_data = datasets.ImageFolder(valid_dir, transform = train_transforms)

	trainloader = torch.utils.data.DataLoader(train_data, batch_size = 50, shuffle = True)

	testloader = torch.utils.data.DataLoader(test_data, batch_size = 50)

	validloader = torch.utils.data.DataLoader(valid_data, batch_size = 30)

	# Defining the model architecture
	if type(args.arch) == type(None) or args.arch == 'vgg16':
		model = models.vgg16(pretrained = True)
		model.name = 'vgg16'
	elif args.arch == 'alexnet':
		model = models.alexnet(pretrained = True)
		model.name = 'alexnet'
	elif args.arch == 'densenet121':
		model = models.densenet121(pretrained = True)
		model.name = 'densenet121'
	else:
		print("Model arch not found")

	for param in model.parameters():
		param.requires_grad = False


	if type(args.hidden_units) == type(None):
		hidden_units = 4096
		print("Number of hidden layers is 4096.")

	if model.name == 'vgg16':
		inputfeat = model.classifier[0].in_features
	elif model.name == 'alexnet':
		inputfeat = model.classifier.in_features
	elif model.name == 'densenet121':
		inputfeat = model.classifier[1].in_features

	# Defining the classifier to be used with the architecture
	classifier = nn.Sequential(OrderedDict([
								('fc1', nn.Linear(inputfeat, hidden_units, bias = True)),
								('relu1', nn.ReLU()),
								('dropout1', nn.Dropout(p = 0.5)),
								('fc2', nn.Linear(hidden_units, 102, bias = True)),
								('output', nn.LogSoftmax(dim = 1))
								]))
	model.classifier = classifier

	# Selecting the device to be used for training the network
	device = torch.device('cuda' if torch.cuda.is_available() and args.processing else 'cpu')
	model.to(device);

	if type(args.learning_rate) == type(None):
		lrate = 0.001
		print("Learning rate specified is 0.001")
	else: lrate = args.lr

	# Defining the loss functions and the optimizer to be used for training
	criterion = nn.NLLLoss()
	optimizer = optim.Adam(model.classifier.parameters(), lr = lrate)


	trained_model = train_network(model, trainloader, validloader, device, criterion, optimizer, args.epochs)

	print("\nTraining process is now complete!")

	# Validate the model for its accuracy
	validate_model(trained_model, testloader, device)

	# Save the model
	save_model(trained_model, args.save_dir, train_data)


if __name__ == '__main__':main()
    