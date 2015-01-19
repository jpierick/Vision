#!/usr/bin/python
# ==============================================================================
# Name:  facerec.py
# Desc:  This program is used to recognize faces from a webcam.  A set of 
#   training images must already exist in subfolders under the "training" 
#   folder (under the folder in which this program exists).  The name of each
#   subfolder under "training" is the name of person that the image represents.
# To Do:  The following changes could be made to make the program more robust:
# * Ensure that the files in the training subfolders are images.
# * Ensure that all the training images are the same format and size.
# * Ensure that each training subfolder has at least one image.
# Change History:
# 2015-02-02 JSP: Created.
# ==============================================================================

from sklearn.decomposition import RandomizedPCA
import numpy as np
import glob
import cv2
import math
import os.path
import string

# Ensure that the training subfolder exists:
if not os.path.isdir('training'):
  print 'The "training" subfolder could not be found.'
  quit(1)

# Ensure that there is at least one training image:
imageCount = 0
for folderName in os.listdir('training'):
  subfolderName = os.path.join('training', folderName)
  if os.path.isdir(subfolderName):
    for fileName in os.listdir(subfolderName):
      if os.path.basename(fileName).isdigit():
        imageCount = imageCount + 1

if imageCount == 0:
  print "Couldn't find any training images."
  quit(1)
    
quit(0)

# Function to get ID from filename
def ID_from_filename(filename):
    part = string.split(filename, '/')
    return part[1].replace("s", "")
 
# Function to convert image to right format
def prepare_image(filename):
    img_color = cv2.imread(filename)
    img_gray = cv2.cvtColor(img_color, cv2.cv.CV_RGB2GRAY)
    img_gray = cv2.equalizeHist(img_gray)
    return img_gray.flat

IMG_RES = 92 * 112 # img resolution
NUM_EIGENFACES = 10 # images per train person
NUM_TRAINIMAGES = 500 # total images in training set

#loading training set from folder train_faces
folders = glob.glob('train_faces/*')
 
# Create an array with flattened images X
# and an array with ID of the people on each image y
X = np.zeros([NUM_TRAINIMAGES, IMG_RES], dtype='int8')
y = []

# Populate training array with flattened imags from subfolders of train_faces and names
c = 0
for x, folder in enumerate(folders):
    train_faces = glob.glob(folder + '/*')
    for i, face in enumerate(train_faces):
        print i
        print ": "
        print face
        print "\n"

        X[c,:] = prepare_image(face)
        y.append(ID_from_filename(face))
        c = c + 1

# perform principal component analysis on the images
pca = RandomizedPCA(n_components=NUM_EIGENFACES, whiten=True).fit(X)
X_pca = pca.transform(X)

# load test faces (usually one), located in folder test_faces
test_faces = glob.glob('test_faces/*')

# Create an array with flattened images X
X = np.zeros([len(test_faces), IMG_RES], dtype='int8')
 
# Populate test array with flattened imags from subfolders of train_faces 
for i, face in enumerate(test_faces):
    X[i,:] = prepare_image(face)
 
# run through test images (usually one)
for j, ref_pca in enumerate(pca.transform(X)):
    distances = []
    # Calculate euclidian distance from test image to each of the known images and save distances
    for i, test_pca in enumerate(X_pca):
        dist = math.sqrt(sum([diff**2 for diff in (ref_pca - test_pca)]))
        distances.append((dist, y[i]))
 
    found_ID = min(distances)[1]

    print "Identified (result: "+ str(found_ID) +" - dist - " + str(min(distances)[0])  + ")"

