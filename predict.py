"""
@author: p0tat0
@title: Image Classifier
"""

# Importing Libraries

import argparse
import json
from math import ceil
import torch
import PIL
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import seaborn as sns
from torchvision import datasets, transforms, models
from torch import nn
from torch import optim
from collections import OrderedDict
from torch.autograd import Variable


# Start of the program




# For Input as Command Line args
def arg_parse():
	# Defining a parser
	parser = argparse.ArgumentParser(description = "Inputs for Neural Networks Attributes")

	# Argument for path of image to be classified
	parser.add_argument('--image', type = str, help = 'Path of image for prediction', required = True)

	# Path of the saved checkpoint of a pretrained network
	parser.add_argument('--checkpoint', type = str, help = 'Path of the saved model', required = True)

	# Specify the max no of types in the graph
	parser.add_argument('--top_k', type = int, help = 'Max classes in the graph')

	#Import category names
	parser.add_argument('--category_names', type = str, help = 'Mapping from categories to real name')

	# Add all the parsed arguments to the variable args
	args = parser.parse_args()

	# Input if gpu or cpu should be used for prediction
    parser.add_argument('--processing', type = str, help = 'Either CPU or GPU')

	# Return the arguments accepted from the user
	return args




# Defination to load a neural network from saved checkpoint
def Load_check(checkpoint_path):

	# Load the saved model
	checkpoint = torch.load(checkpoint_path)

	# Load Defaults if nothing is loaded
	if checkpoint['architecture'] == 'alexnet':
		model = models.alexnet(pretrained=True)
	elif checkpoint['architecture'] == 'vgg16':
		model = models.vgg16(pretrained=True)
	elif checkpoint['architecture'] == 'densenet121':
		model = models.densenet121(pretrained=True)
	else:
		print("Model arch not found.")
	# Freeze parameters to stop backpropagation
	for param in model.parameters():
		param.requires_grad = False

	# Load attributes of the Neural Network
	model.class_to_idx = checkpoint['class_to_idx']
	model.classifier = checkpoint['classifier']
	model.load_state_dict(checkpoint['state_dict'])

	# Return the loaded model
	return model



# Definition for preprocessing the images
def Process_image(image_path):

	image = PIL.Image.open(image_path)

	# Declaring the transforms to resize the image
	transform = transforms.Compose([transforms.Resize(255),
									transforms.CenterCrop(225),
									transforms.ToTensor(),
									transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                                         std = [0.229, 0.224, 0.225])])
	# Apply the transformations to the image
	final_image = transform(image)

	# Return the final image
	return final_image



# Definition for prediction of the given image using the neural network
def Predict(image, model, cat_to_name, top_k = 5, device):
	''' Predict the class of image using a trainde deep learning model

	image: It is the processed image
	model: It is the trained neural network
	device: The device to be used for prediction. Gpu's work faster
	top_k: The top classes to be considered

	'''

	# Setting the model to evaluate mode so that it dose'nt trains of the given image
	model.eval();

	# Transferring the model to cpu
	model = model.device()
	# Processing image as inputs and running through the model
	output = model.forward(Variable(image.unsqueeze(0), volatile = True))
	output = torch.exp(output)

	probs, classes = output.topk(top_k)
	probs = probs.exp().data.numpy()[0]
	classes = classes.data.numpy()[0]
	class_keys = {x: y for y, x in model.class_to_idx.items()}
	classes = [class_keys[i] for i in classes]

	
	flowers = [cat_to_name[i] for i in classes]
	return flowers, probs, classes


# Definition to print probability
def Print_prob(probs, flowers):
    """
    Converts two lists into a dictionary to print on screen
    """
    
    for i, j in enumerate(zip(flowers, probs)):

        print ("Rank {}:".format(i),
               "Flower: {}, liklihood: {}%".format(j[1], ceil(j[0]*100)))


# Main Function
def main():
	'''
	Main execution function
	'''
	args = arg_parse()

	with open(args.category_names, 'r') as f:
		cat_to_name = json.load(f)
	model = Load_check(args.checkpoint)

	image = Process_image(args.image)

	device = torch.device('cuda' if torch.cuda.is_available() and args.processing else 'cpu')
	flowers, prob, labels = Predict(image, model, cat_to_name, args.top_k, device)
	nn.log_softmax(prob)
	Print_prob(flowers, prob)

if __name__ ==  '__main__': main()