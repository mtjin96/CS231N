import numpy as np
import matplotlib.pyplot as plt
import sys
#import torchvision.models as models
import cv2
import torch
import torch.nn as nn
#import torchvision.transforms as transforms
#import torchvision.datasets as datasets
import glob
import imp

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


def run_model(modelname):
	total = 256
	MainModel = imp.load_source('MainModel', 'senet50_256_pytorch/senet50_256_pytorch.py')
	model = torch.load('senet50_256_pytorch/senet50_256_pytorch.pth')
	model.cuda()
	model.eval()

	# run_and_save('Train', total, model, modelname)
	run_and_save('Test', total, model, modelname)
	# run_and_save('Valid', total, model, modelname)

def run_and_save(dataset, total, model, modelname):
	dirname = ''
	startInt = 0
	if dataset == 'Train':
		dirname = '../images/Train_crop/*/'
		startInt = 37
		savename = '../images/Train_' + modelname +'_feature/'
	elif dataset == 'Test':
		dirname = '../images/Test_crop/*/'
		startInt = 35
		savename = '../images/Test_'+ modelname + '_feature/'
	elif dataset == 'Valid':
		dirname = '../images/Valid_crop/*/'
		startInt = 37
		savename = '../images/Valid' + modelname + '_feature/'
	list_of_dirs = sorted(glob.glob(dirname))
	for dir in list_of_dirs:
		# print(dir[21:-1])
		# exit()
		list_of_files = sorted(glob.glob(dir+'*.jpg'), key=lambda x: int(x[startInt:-9]))
		inputs = []
		if dataset == 'Test':
			filename = 'I' + dir[20:-1] + '_image.ssv'	# test_crop
		else:
			filename = dir[21:-1] + '_image.ssv'
		print(filename)
		file = open(savename+filename, 'w')
		file.write('Frametime ')

		for i in range(total):
			file.write('vector' + str(i) + ' ')
		file.write('\n')

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
			inputs.append(torch.tensor(im, device='cuda:0'))

		inputs = torch.stack(inputs)

		pred_list = []
		for data in torch.split(inputs, 1, 0):
			torch.cuda.empty_cache()
			with torch.no_grad():
				pred = model(data)
				# print("pred shape: ", pred.shape)
				pred = pred.view(1, pred.shape[1])
			pred_list.append(pred)
		pred_list = torch.stack(pred_list, dim=0)
		pred_list = torch.squeeze(pred_list, dim=1)
		pred_list = pred_list.to(torch.device("cpu")).numpy()	# ->(384, 1000)
		time = 0.01
		for line in pred_list:
			file.write(str(time)+' ')
			time += 0.5
			for vec in line:
				file.write(str(vec) + ' ')
			file.write('\n')

		file.close()


if __name__ == "__main__":
	#run_model('vgg16')
	#run_model('resnet')
	run_model('vggFace2')


# im = cv2.imread('frame0.jpg')
# im = cv2.resize(im, (224, 224)).astype(np.float32)
# scaled_im = []
# for i in im:
#   ilist = []
#   for j in i:
#       jlist = []
#       for k in j:
#           jlist.append(float(k)/255.0)
#       ilist.append(jlist)
#   scaled_im.append(ilist)

# scaled_im = np.asarray(scaled_im).astype(np.float32)


# im[:,:,0] -= 103.939
# im[:,:,1] -= 116.779
# im[:,:,2] -= 123.68
# scaled_im = scaled_im.transpose((2,0,1))
# scaled_im = np.expand_dims(scaled_im, axis=0)
# scaled_im = torch.tensor(scaled_im)
# print(scaled_im)


# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                  std=[0.229, 0.224, 0.225])

# image = datasets.ImageFolder('Test_resized/', transforms.Compose([
#             transforms.ToTensor(),
#             normalize,
#         ]))


# print('loading data - done')
# print(sum(num_image))

# begin_index = 0
# for i in num_image:
#   test_images = []
#   for j in range(begin_index, begin_index+i):
#       test_images.append(image[j][0])
#   begin_index += i

#   test_images = torch.stack(test_images)
#   print(test_images.shape)
#   pred_list = []
#   for data in torch.split(test_images, 1, 0):
#       pred = vgg16(data)
#       pred_list.append(pred)

#   print(len(pred_list))
#   print(pred_list[0].shape)
#   print(pred_list[9].shape)




# im = torch.unsqueeze(image[0][0], dim=0)
# im2 = torch.unsqueeze(image[1][0], dim=0)
# a = [im, im2]
# a = torch.stack(a)
# a = torch.squeeze(a, dim=1)


# pred = vgg16(a)
# print(pred)
# print(pred.shape)



# print(scaled_im)
# output = vgg16(scaled_im)
# print(output)
