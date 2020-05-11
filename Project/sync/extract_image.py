import cv2
import glob
import os
# from vgg16 import VGG16
# import torch
import numpy as np
import shutil

''' video to image '''
# list_of_files = glob.glob('Valid/*.mp4')
# for file in list_of_files:
# 	filename = file[6:16]
# 	# print(filename)
# 	if not os.path.exists("Images/Valid"+filename):
# 		os.makedirs("Images/Valid/"+filename)
# 	vidcap = cv2.VideoCapture(file)
# 	success,image = vidcap.read()
# 	count = 0
# 	success = True
# 	while success:
# 		if count%15 == 0:
# 			cv2.imwrite("Images/Valid/"+filename+"/frame%d.jpg" % count, image)     # save frame as JPEG file
# 			print ('Read a new frame: ', success)
# 		success,image = vidcap.read()
# 		count += 1

# def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
# 	# initialize the dimensions of the image to be resized and
# 	# grab the image size
# 	dim = None
# 	(h, w) = image.shape[:2]
# 	print(image.shape)

# 	# if both the width and height are None, then return the
# 	# original image
# 	if width is None and height is None:
# 		return image

# 	# check to see if the width is None
# 	if width is None:
# 		# calculate the ratio of the height and construct the
# 		# dimensions
# 		r = height / float(h)
# 		dim = (int(w * r), height)

# 	# otherwise, the height is None
# 	else:
# 		# calculate the ratio of the width and construct the
# 		# dimensions
# 		r = width / float(w)
# 		dim = (width, int(h * r))

# 	# resize the image
# 	resized = cv2.resize(image, dim, interpolation = inter)

# 	# return the resized image
# 	return resized


# #load cascade classifier training file for haarcascade
# haar_face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_alt.xml')

# #convert the test image to gray image as opencv face detector expects gray images
# def detect_faces(f_cascade, colored_img, scaleFactor = 1.3):
# 	img_copy = np.copy(colored_img)
# 	#convert the test image to gray image as opencv face detector expects gray images
# 	gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
# 	#let's detect multiscale (some images may be closer to camera than others) images
# 	faces = f_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=5);
# 	#go over list of faces and draw them as rectangles on original colored img
# 	return faces

# def corp_faces(imagePath):
# 	# print( "Process image: " + imagePath)
# 	input_image = cv2.imread(imagePath)
# 	faces = detect_faces(haar_face_cascade, input_image)
# 	if faces == ():
# 		return 0, 0, 0, 0
# 	for (x, y, w, h) in faces:
# 		return x, y, w, h


# ''' read image '''
# list_of_dirs = glob.glob('images/Train/*/')
# for dir in list_of_dirs:
# 	print(dir)
# 	if 'ID129_vid' in dir:
# 		list_of_files = glob.glob(dir+'*.jpg')
# 		# xlist, ylist, wlist, hlist = [], [], [], []
# 		# for file in list_of_files:
# 		# 	if 'face' not in file:
# 		# 		x, y, w, h = corp_faces(file)
# 		# 		if x != 0:
# 		# 			xlist.append(x)
# 		# 			ylist.append(y)
# 		# 			wlist.append(w)
# 		# 			hlist.append(h)

# 		# xavg = int(sum(xlist)/float(len(xlist)))
# 		# yavg = int(sum(ylist)/float(len(ylist)))
# 		# wavg = int(sum(wlist)/float(len(wlist)))
# 		# havg = int(sum(hlist)/float(len(hlist)))


# 		for file in list_of_files:
# 			# ypadding = int((224 - havg)/2)
# 			# xpadding = int((224 - wavg)/2)
# 			if 'face' not in file:
# 				# if yavg+havg+ypadding < 0 or yavg-ypadding < 0:
# 				# 	crop_img = cv2.imread(file)[0:224, xavg-xpadding:xavg+wavg+xpadding]
# 				# else:
# 				crop_img = cv2.imread(file)[50:200, 180:330]
# 				# crop_img = cv2.imread(file)[0:224, 150:374]	# ID 156
# 				# crop_img = cv2.imread(file)[0:224, 150:374]	# ID 144
# 				# crop_img = cv2.imread(file)[0:224, 150:374]	# ID 141
# 				# crop_img = cv2.imread(file)[0:224, 110:334]	# ID 142
# 				# crop_img = cv2.imread(file)[0:224, 110:334]	# ID 149
# 				# crop_img = cv2.imread(file)[0:224, 160:384]	# ID 121
# 				# crop_img = cv2.imread(file)[30:254, 150:374]	# ID 129
# 				outputPath = file[:-4] + "_face.jpg"
# 				print(outputPath)
# 				cv2.imwrite(outputPath, crop_img)


''' copy file '''
# list_of_dirs = glob.glob('images/Train_crop/*/')
# for dir in list_of_dirs:
# 	newdir = dir[13:]
# 	if not os.path.exists("images/Train_resized/"+newdir):
# 		os.makedirs("images/Train_resized/"+newdir)
# 	list_of_files = glob.glob(dir+'*.jpg')
# 	for file in list_of_files:
# 		if 'resized' in file:
# 			shutil.move(file, "images/Train_resized/"+newdir)			

''' delete file '''
# list_of_dirs = glob.glob('images/Train_resized/*/')
# for dir in list_of_dirs:
# 	list_of_files = glob.glob(dir+'*.jpg')
# 	for file in list_of_files:
# 		if 'resresized' in file:
# 			os.remove(file)



# # ''' read image and feed into model '''
# # list_of_files = glob.glob('images/Train_crop/ID111_vid4/*.jpg')
# # for file in list_of_files:
# # 	im = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
# # 	# im = cv2.imread(file)
# # 	# print(im)
# # 	# (480, 270)

# # im = cv2.resize(im, (100, 100)).astype(np.float32)
# # im = np.expand_dims(im, axis=0)
# # # im[:,:,0] -= 103.939
# # # im[:,:,1] -= 116.779
# # # im[:,:,2] -= 123.68
# # # im = im.transpose((2,0,1))
# # im = np.expand_dims(im, axis=0)


# # # (1, 1, 224, 224)

# # model = VGG16(128)
# # im = torch.tensor(im)
# # predict = model(im)



list_of_dirs = sorted(glob.glob('images/Train_crop/*/'))
mean1 = np.zeros((50,50))
n1 = 0
for dir in list_of_dirs:
	list_of_files = sorted(glob.glob(dir+'*.jpg'), key=lambda x: int(x[34:-9]))
	time = 0.01
	for image in list_of_files:
		im = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
		im = cv2.resize(im, (50,50))
		im = im.astype(np.float32)
		mean1 += im
		n1 += 1

mean1 /= n1
print('finished mean')


# ''' generate csv files for images 
# 	np.reshape(flattened_image, (224, 224, 3)) '''
list_of_dirs = sorted(glob.glob('images/Test_crop/*/'))
for dir in list_of_dirs:
	print(dir)
	filename = dir[17:-1] + '_image.ssv'
	file = open('images/Test_ssv/'+filename, 'w')
	file.write('Frametime ')
	for i in range(50*50):
		file.write('pixel' + str(i) + ' ')
	file.write('\n')
	list_of_files = sorted(glob.glob(dir+'*.jpg'), key=lambda x: int(x[32:-9]))   # train(34,-9), test(32,-9)
	time = 0.01
	for image in list_of_files:
		im = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
		im = cv2.resize(im, (50,50))
		im = im.astype(np.float32)
		im -= mean1
		im = im.flatten()
		file.write(str(time)+' ')
		time += 0.5
		i = 0
		for n in im:
			file.write(str(n) + ' ')
			i+=1

		file.write('\n')

	file.close()





# list_of_dirs = glob.glob('images/Train_crop/*/')
# for dir in list_of_dirs:
# 	print(dir)
# 	list_of_files = glob.glob(dir+'*.jpg')
# 	for file in list_of_files:
# 		im = cv2.imread(file)
# 		im = cv2.resize(im, (224,224))
# 		filename = file[:-8] + 'resized.jpg'
# 		cv2.imwrite(filename,im)