from imutils.object_detection import non_max_suppression
from sklearn.svm import *
from skimage.feature import hog
from skimage.transform import pyramid_gaussian
from skimage import color
import imutils
import os
import glob
import joblib
from google.colab.patches import cv2_imshow
import pickle
import itertools
import glob
import os.path
from PIL import Image
import numpy as np
import cv2

# orientations = 9
# pixels_per_cell = (8, 8)
# cells_per_block = (2, 2)
# threshold = .3

def make_training_data(orientations = 9,pixels_per_cell = (8, 8),cells_per_block = (2, 2),threshold = .3,img_address="/content/drive/MyDrive/CV/seq03-img-left/*",annotation_address="./bahnhof-annot.idl"):
	file1 = open("./bahnhof-annot.idl","r+") 
	x = file1.read()
	ann = {}
	for it in x.split(";"):
	    ind = it.split(":")[0].replace("\n","").strip("\"")
	    ann[ind] = []
	    for y in it.split(":")[-1].split("),"):
	        y = y.strip().strip(".").strip("(").strip(")")
	        li = []
	        for yy in y.split(","):
	            li.append(int(yy.strip()))
	        ann[ind].append(li)
	file1.close()

	image_features = []
	target = []

	print("Computing positive HoG features")
	for fname in glob.glob('/content/drive/MyDrive/CV/seq03-img-left/*'):
	    I = Image.open(fname)
	    key = "left/"+fname.split("/")[-1]
	    if(key in ann):
	      for (x1,y1,x2,y2) in ann["left/"+fname.split("/")[-1]]:
	        if(x1>x2):
	          t = x1
	          x1 = x2
	          x2 = t
	        if(y1>y2):
	          t = y1
	          y1 = y2
	          y2 = t
	        img = I.crop((x1,y1,x2,y2))
	        img = img.resize((64,128))
	        gray= img.convert('L')
	        fd = hog(gray, orientations, pixels_per_cell, cells_per_block, block_norm='L2', feature_vector=True) 
	        image_features.append(fd)
	        target.append(1)

	print("Computing negative HoG features")
	for fname in glob.glob('/content/drive/MyDrive/CV/seq03-img-left/*'):
	    Is =cv2.imread(fname)
	    I = Image.open(fname)
	    sample2 = np.random.randint(0,Is.shape[0]-128,5)
	    sample1 = np.random.randint(0,Is.shape[1]-64,5)
	    samples = zip(sample1,sample2)
	    for j in samples:
	      I1 = I.crop((j[1],j[0],j[1]+128,j[0]+64))
	      I1 = I1.resize((64,128))
	      gray= I1.convert('L')
	      fd = hog(gray, orientations, pixels_per_cell, cells_per_block, block_norm='L2', feature_vector=True) 
	      image_features.append(fd)
	      target.append(0)

	X = np.asarray(image_features,dtype=np.float64)
	Y = np.asarray(target,dtype= np.float64)
	X = np.reshape(X, (X.shape[0],X.shape[1]))

  return X,Y