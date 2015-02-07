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

# Read through all of the training images to determine their dimensions and
# to ensure that there is at least one training image.  If any of the images
# have different dimensions than the others, then report the error and quit:
imageCount = 0
height = 0
width = 0
depth = 0
minImagesPerFolder = 0
maxImagesPerFolder = 0
for folderName in os.listdir('training'):
  imagesInFolder = 0
  subfolderName = os.path.join('training', folderName)
  if os.path.isdir(subfolderName):
    for fileName in os.listdir(subfolderName):
      if os.path.splitext(os.path.basename(fileName))[0].isdigit():
        # Keep track of the number of training images:
        imageCount = imageCount + 1
        imagesInFolder = imagesInFolder + 1

        # Read the image so that it can be processed:
        img = cv2.imread(os.path.join(subfolderName, fileName))
        if img == None: 
          print "Unable to read image " + os.path.join(subfolderName, fileName)
          quit(1)

        # Determine the images dimensions and ensure that they are the same
        # as all previous training images that have been read:
        imgHeight, imgWidth, imgDepth = img.shape
        if height == 0:
          height = imgHeight
          width = imgWidth
          depth = imgDepth
        elif imgHeight <> height or imgWidth <> width or imgDepth <> depth:
          print "File " + fileName + " in folder " + subfolderName + \
            " has different dimensions than files previously read."
          quit(1)

  # Track the minimum and maximum number of training images in all the 
  # folders:
  if minImagesPerFolder == 0:
    minImagesPerFolder = imagesInFolder
  elif imagesInFolder < minImagesPerFolder:
    minImagesPerFolder = imagesInFolder

  if imagesInFolder > maxImagesPerFolder:
    maxImagesPerFolder = imagesInFolder

if imageCount == 0:
  print "Couldn't find any training images."
  quit(1)

print "Mininum number of training images for a subject: " + str(minImagesPerFolder)
print "Maximum number of training images for a subject: " + str(maxImagesPerFolder)

# Create an array of flattened training images and a corresponding array of
# names (the name associated with each corresponding training image):
trainingImages = np.zeros([imageCount, height * width], dtype='int8')
trainingName = []

# Populate the arrays with the flattened training images:
imageNumber = 0
for folderName in os.listdir('training'):
  subfolderName = os.path.join('training', folderName)
  if os.path.isdir(subfolderName):
    for fileName in os.listdir(subfolderName):
      if os.path.splitext(os.path.basename(fileName))[0].isdigit():
        print "Processing image " + fileName + " for " + folderName

        # Track the name associated with the training image:
        trainingName.append(folderName)

        # Read the image so that it can be processed:
        img = cv2.imread(os.path.join(subfolderName, fileName))
        if img == None: 
          print "Unable to read image " + os.path.join(subfolderName, fileName)
          quit(1)

        img = cv2.cvtColor(img, cv2.cv.CV_RGB2GRAY)
        trainingImages[imageNumber,:] = img.flat
        imageNumber = imageNumber + 1
    
# Perform principal component analysis on the images:
print "Performing principal component analysis."
pca = RandomizedPCA(n_components=maxImagesPerFolder, whiten=True).fit(trainingImages)
pcaTrainingImages = pca.transform(trainingImages)

quit(0)

