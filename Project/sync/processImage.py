# /imagePairs/pair1_M/consImage_0.jpg

#import required libraries 
import numpy as np
# import cv2
import matplotlib.pyplot as plt
import time 
import os 

# def convertToRGB(img):
#     return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
#     # initialize the dimensions of the image to be resized and
#     # grab the image size
#     dim = None
#     (h, w) = image.shape[:2]

#     # if both the width and height are None, then return the
#     # original image
#     if width is None and height is None:
#         return image

#     # check to see if the width is None
#     if width is None:
#         # calculate the ratio of the height and construct the
#         # dimensions
#         r = height / float(h)
#         dim = (int(w * r), height)

#     # otherwise, the height is None
#     else:
#         # calculate the ratio of the width and construct the
#         # dimensions
#         r = width / float(w)
#         dim = (width, int(h * r))

#     # resize the image
#     resized = cv2.resize(image, dim, interpolation = inter)

#     # return the resized image
#     return resized

# #load cascade classifier training file for haarcascade
# haar_face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_alt.xml')

# #convert the test image to gray image as opencv face detector expects gray images
# def detect_faces(f_cascade, colored_img, scaleFactor = 1.3):
#     img_copy = np.copy(colored_img)
#     #convert the test image to gray image as opencv face detector expects gray images
#     gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
#     #let's detect multiscale (some images may be closer to camera than others) images
#     faces = f_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=5);
#     #go over list of faces and draw them as rectangles on original colored img
#     return faces

# def corp_faces(imagePath):
#     print "Process image: " + imagePath
#     input_image = cv2.imread(imagePath)
#     faces = detect_faces(haar_face_cascade, input_image)
#     for (x, y, w, h) in faces:
#         padding = 15
#         crop_img = input_image[y-padding:y+h+padding, x-padding:x+w+padding]
#         # img_yuv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2YUV)
#         # # equalize the histogram of the Y channel
#         # img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
#         # # convert the YUV image back to RGB format
#         # img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
#         outputPath = imagePath[:-4] + "_face.jpg"
#         img_output = image_resize(crop_img, height = 300)
#         cv2.imwrite(outputPath,img_output)
        

# def processImage(dir):                                                                                                  
#     r = []                                                                                                            
#     subdirs = [x[0] for x in os.walk(dir)]                                                                            
#     for subdir in subdirs:                                                                                            
#         files = os.walk(subdir).next()[2]                                                                             
#         if (len(files) > 0):                                                                                          
#             for file in files:
#                 if file[-1] == 'e':
#                     continue
#                 if file[-5] != 'e':
#                     imagePath = subdir + "/" + file  
#                     #load test iamge
#                     corp_faces(imagePath)

def cleanImage(dir):
    r = []
    count = 0                                                                                                            
    subdirs = [x[0] for x in os.walk(dir)]
    printBuffer = ['', '']                                                                          
    for subdir in subdirs:                                                                                            
        files = os.walk(subdir).next()[2]                                                                             
        if (len(files) > 0):                                                                                          
            for file in files: 
                imagePath = subdir + "/" + file  
                if file[-5] != 'e':
                    os.remove(imagePath)
                    # fileTmp = "consI" + file[1:-9] + ".jpg"
                    # # print fileTmp
                    # if fileTmp in files:
                    #     printBuffer[0] = imagePath[1:]
                    # else:
                    #     printBuffer[1] = imagePath[1:]
                    # if count%2 == 1:
                    #     print "[\"" + printBuffer[0] + "\",\"" + printBuffer[1] + "\"],"
                    #     printBuffer = ['', ''] 
                    # count = count + 1

def renameImage(dir):
    r = []
    count = 0                                                                                                            
    subdirs = [x[0] for x in os.walk(dir)]
    printBuffer = []                                                                          
    for subdir in subdirs:                                                                                            
        files = os.walk(subdir).next()[2]                                                                             
        if (len(files) > 0):                                                                                          
            for file in files: 
                imagePath = subdir + "/" + file  
                if file[-5] == 'e':
                    # os.remove(imagePath)
                    newName = "image" + "_" + "_".join(file.split("_")[1:])
                    print newName
                    os.rename(subdir + "/" + file,  subdir + "/" + newName)

# renameImage("./imagePairs")
cleanImage("./imagePairs")
