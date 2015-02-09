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
import cv2
import glob
import math
import numpy as np
import os.path
import string
import sys
import time

# Ensure that the training subfolder exists:
if not os.path.isdir('training'):
  print 'The "training" subfolder could not be found.'
  quit(1)

# Read through all of the training images to determine their dimensions and
# to ensure that there is at least one training image.  If any of the images
# have different dimensions than the others, then report the error and quit:
imgTrainingCountTotal = 0
height = 0
width = 0
depth = 0
minImagesPerFolder = 0
maxImagesPerFolder = 0
for folderName in os.listdir('training'):
  imgCountInFolder = 0
  folderFullName = os.path.join('training', folderName)
  if os.path.isdir(folderFullName):
    for fileName in os.listdir(folderFullName):
      if os.path.splitext(os.path.basename(fileName))[0].isdigit():
        # Keep track of the number of training images:
        imgTrainingCountTotal = imgTrainingCountTotal + 1
        imgCountInFolder = imgCountInFolder + 1

        # Read the image so that it can be processed:
        img = cv2.imread(os.path.join(folderFullName, fileName))
        if img == None: 
          print "Unable to read image " + os.path.join(folderFullName, fileName)
          quit(1)

        # Determine the images dimensions and ensure that they are the same
        # as all previous training images that have been read:
        imgHeight, imgWidth, imgDepth = img.shape
        if height == 0:
          height = imgHeight
          width = imgWidth
          depth = imgDepth
        elif imgHeight <> height or imgWidth <> width or imgDepth <> depth:
          print "File " + fileName + " in folder " + folderFullName + \
            " has different dimensions than files previously read."
          quit(1)

  # Track the minimum and maximum number of training images in all the 
  # folders:
  if minImagesPerFolder == 0:
    minImagesPerFolder = imgCountInFolder
  elif imgCountInFolder < minImagesPerFolder:
    minImagesPerFolder = imgCountInFolder

  if imgCountInFolder > maxImagesPerFolder:
    maxImagesPerFolder = imgCountInFolder

if imgTrainingCountTotal == 0:
  print "Couldn't find any training images."
  quit(1)

print "Mininum number of training images for a subject: " + str(minImagesPerFolder)
print "Maximum number of training images for a subject: " + str(maxImagesPerFolder)

# Create an array of flattened training images and a corresponding array of
# names (the name associated with each corresponding training image):
trainingImages = np.zeros([imgTrainingCountTotal, height * width], dtype='int8')
trainingName = []

# Populate the arrays with the flattened training images:
imageNumber = 0
for folderName in os.listdir('training'):
  folderFullName = os.path.join('training', folderName)
  if os.path.isdir(folderFullName):
    for fileName in os.listdir(folderFullName):
      if os.path.splitext(os.path.basename(fileName))[0].isdigit():
        print "Processing image " + fileName + " for " + folderName

        # Track the name associated with the training image:
        trainingName.append(folderName)

        # Read the image so that it can be processed:
        img = cv2.imread(os.path.join(folderFullName, fileName))
        if img == None: 
          print "Unable to read image " + os.path.join(folderFullName, fileName)
          quit(1)

        img = cv2.cvtColor(img, cv2.cv.CV_RGB2GRAY)
        trainingImages[imageNumber,:] = img.flat
        imageNumber = imageNumber + 1
    
# Perform principal component analysis on the images:
print "Performing principal component analysis of training images."
pca = RandomizedPCA(n_components=maxImagesPerFolder, whiten=True).fit(trainingImages)
pcaTrainingImages = pca.transform(trainingImages)


# Verify that the cascade file exists that will be used to detect faces in the
# webcam feed:
cascPath = './haarcascade_frontalface_default.xml'
if not os.path.isfile(cascPath):
  print 'The face recognition cascade file (' + cascPath + ') does not exist.'
  quit(1)

faceCascade = cv2.CascadeClassifier(cascPath)

# Specify which camera to use to capture the video stream from which faces
# will be detected:
video_capture = cv2.VideoCapture(0)

while True:
  # On each loop, we check to see if any faces were detected in the image.
  # If one (or more) was, then the user is given the opportunity to keep
  # the image (the grayscale version) as a training image:
  face_found = False

  ret, frame = video_capture.read()
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),
    flags=cv2.cv.CV_HAAR_SCALE_IMAGE
  )

  # Put a green box around all the faces that were found in the image:
  for (x, y, w, h) in faces:
    cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
    face_found = True

  cv2.imshow('Video', frame)
  key = cv2.waitKey(1) & 0xFF
  if key == ord('q'):
    break

  if face_found == True:
    for (x, y, w, h) in faces:
      face_img = gray[y:y+h, x:x+w]
      face_img = cv2.resize(face_img, (100,100), interpolation=cv2.INTER_CUBIC)
      face_img = cv2.equalizeHist(face_img)

      testImages = np.zeros([1, 100*100], dtype='int8')
      testImages[0,:] = face_img.flat

      for j, ref_pca in enumerate(pca.transform(testImages)):
        distances = []
        for i, test_pca in enumerate(pcaTrainingImages):
          dist = math.sqrt(sum([diff**2 for diff in (ref_pca - test_pca)]))
          distances.append((dist, trainingName[i]))
  
        found_ID = min(distances)[1]
        print "Identified (result: " + found_ID + " - dist - " + str(min(distances)[0]) + ")"

        face_found = False

video_capture.release()
cv2.destroyAllWindows()

