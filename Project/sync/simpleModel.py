import torch
from torch.utils.data import DataLoader
from torch.utils.data import sampler
from torch.utils.data import TensorDataset
import glob
import cv2
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class Flatten(nn.Module):
	def forward(self, x):
		return flatten(x)

def flatten(x):
	N = x.shape[0] # read in N, C, H, W
	return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image


def train_part34(model, optimizer, loader_train, loader_val, epochs=9999):
	
	device = 'cuda:0'
	dtype = torch.float
	model = model.to(device=device)  # move the model parameters to CPU/GPU
	for e in range(epochs):
		for t, (x, y) in enumerate(loader_train):
			model.train()  # put model to training mode
			x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
			y = y.to(device=device, dtype=torch.float).long()

			scores = model(x)
			# print(scores.shape)
			loss = F.cross_entropy(scores, y)

			# Zero out all of the gradients for the variables which the optimizer
			# will update.
			optimizer.zero_grad()

			# This is the backwards pass: compute the gradient of the loss with
			# respect to each  parameter of the model.
			loss.backward()

			# Actually update the parameters of the model using the gradients
			# computed by the backwards pass.
			optimizer.step()

			if t % 100 == 0:
				print('Iteration %d, loss = %.4f' % (t, loss.item()))
				check_accuracy_part34(loader_val, model)
				print()


def check_accuracy_part34(loader, model):
	# if loader.dataset.train:
	#     print('Checking accuracy on validation set')
	# else:
	#     print('Checking accuracy on test set')  
	device = 'cuda:0'
	dtype = torch.float 
	num_correct = 0
	num_samples = 0
	model.eval()  # set model to evaluation mode
	with torch.no_grad():
		for (x, y) in loader:
			x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
			y = y.to(device=device, dtype=torch.float).long()
			scores = model(x)
			_, preds = scores.max(1)
			num_correct += (preds == y).sum()
			num_samples += preds.size(0)
		acc = float(num_correct) / num_samples
		print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))


def get_rating(dataset):
	if dataset == 'Train':
		dirname = '../data/ratings/Train/classification/*.csv'
	elif dataset == 'Test':
		dirname = '../data/ratings/Test/classification/*.csv'

	list_of_dirs = sorted(glob.glob(dirname))
	ratings = []
	length = []
	for file in list_of_dirs:
		rating_inside = []
		with open(file) as f:
			line = f.readlines()
			line = line[1:]
			length.append(len(line))
			for l in line:
				rating_inside.append(float(l.split(',')[1].strip()))
		ratings.append(rating_inside)
	return ratings

def get_data(dataset):

	dirname = ''
	startInt = 0
	if dataset == 'Train':
		dirname = '../images/Train_crop/*/'
		startInt = 37

	elif dataset == 'Test':
		dirname = '../images/Test_crop/*/'
		startInt = 35
	elif dataset == 'Valid':
		dirname = '../images/Valid_crop/*/'
		startInt = 37
	list_of_dirs = sorted(glob.glob(dirname))
	ret = []
	length = []
	for dir in list_of_dirs:
		# print(dir[21:-1])
		# exit()
		list_of_files = sorted(glob.glob(dir+'*.jpg'), key=lambda x: int(x[startInt:-9]))

		length.append(len(list_of_files))
		
		ret_inside = []
		if dataset == 'Test':
			filename = 'I' + dir[20:-1] + '_image.ssv'	# test_crop
		else:
			filename = dir[21:-1] + '_image.ssv'
		print(filename)

		for image in list_of_files:
			im = cv2.imread(image).astype(np.float32)
			im = cv2.resize(im, (224,224))
			im /= 255
			im[:,:,0] -= 0.485
			im[:,:,0] /= 0.229
			im[:,:,1] -= 0.456
			im[:,:,1] /= 0.224
			im[:,:,2] -= 0.406
			im[:,:,2] /= 0.225
			im = im.transpose((2,0,1))
			ret_inside.append(im)
		ret.append(ret_inside)

	return ret

def anti_pad(testdata, test_rating):
	newtestdata = []
	newtestrating = []
	for i in range(len(testdata)):
		minval = min(len(testdata[i]), len(test_rating[i]))
		testdata[i] = testdata[i][:minval]
		test_rating[i] = test_rating[i][:minval]

	for i in range(len(testdata)):
		for j in range(len(testdata[i])):
			newtestdata.append(testdata[i][j])
			newtestrating.append(test_rating[i][j])

	testdata = np.array(newtestdata)
	test_rating = np.array(newtestrating)

	return testdata, test_rating



traindata = get_data('Train')
testdata = get_data('Test')

train_rating = get_rating('Train')
test_rating = get_rating('Test')


train_data, train_rating = anti_pad(traindata, train_rating)	# (N, 3, 224, 224), (N)
test_data, test_rating = anti_pad(testdata, test_rating)

# print(train_data.shape, train_rating.shape)
# print(test_data.shape, test_rating.shape)


model = None
optimizer = None

# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

learning_rate = 1e-3 #1e-2
filter_sz1 = 3
filter_sz2 = 5
filter_sz3 = 3
filter_num1 = 64
filter_num2 = 32
filter_num3 = 16
filter_num4 = 8

layer1 = nn.Sequential(
	# nn.Flatten()
	nn.Conv2d(3, filter_num1, filter_sz1, padding=1),
	nn.BatchNorm2d(filter_num1),
	nn.ReLU(),
	nn.MaxPool2d(2),
	nn.Conv2d(filter_num1, filter_num2, filter_sz2, padding=2),
	nn.BatchNorm2d(filter_num2),
	nn.ReLU(),
	nn.MaxPool2d(2),
	nn.Conv2d(filter_num2, filter_num3, filter_sz3, padding=1),
	nn.BatchNorm2d(filter_num3),
	nn.ReLU(),
	nn.MaxPool2d(2),
	nn.Conv2d(filter_num3, filter_num4, filter_sz3, padding=1),
	nn.BatchNorm2d(filter_num4),
	nn.ReLU(),
	nn.MaxPool2d(2),
	nn.Conv2d(filter_num4, filter_num4, filter_sz3, padding=1),
	nn.BatchNorm2d(filter_num4),
	nn.ReLU(),
	nn.MaxPool2d(2),
)

layer2 = nn.Sequential(
	Flatten(),
	nn.Linear(392, 128),
	nn.BatchNorm1d(128),
	nn.ReLU(),
	nn.Linear(128, 5),
	nn.BatchNorm1d(5)
)


model = nn.Sequential(layer1, layer2)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
################################################################################
#                                 END OF YOUR CODE                             
################################################################################
train = TensorDataset(torch.Tensor(train_data), torch.Tensor(train_rating))
train_loader = DataLoader(train, batch_size = 64, shuffle = True)

test = TensorDataset(torch.Tensor(test_data), torch.Tensor(test_rating))
test_loader = DataLoader(test, batch_size = 64, shuffle = True)


# You should get at least 70% accuracy
train_part34(model, optimizer, train_loader, test_loader, epochs=9999)
			
