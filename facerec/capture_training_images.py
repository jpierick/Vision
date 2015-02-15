#!/usr/bin/python
# ==============================================================================
# Name: capture_training_images.py
# Desc: This program is used to capture training images that will be used by
#   the face recognition program.  The program displays a window with the live
#   video feed.  Any faces that are detected are showed surrounded by a green
#   box.  When the user presses the "k" key, the current face is saved as a
#   unique image in a folder under the "training" folder.
# Change History:
# 2015-02-01 JSP: Created.
# 2015-02-14 JSP: Added configuration file.  Removed histogram equalization of
#   images; this will be done (if desired) in the facerec.py program when the
#   training images are read.
# ==============================================================================

import ConfigParser
import cv2
from decimal import *
import os.path
import sys
import time

# Read the configuration file:
config = ConfigParser.SafeConfigParser()
config.read('facerec.conf')

imgHeight = Decimal(config.get('Image Resolution', 'imgHeight'))
imgWidth = Decimal(config.get('Image Resolution', 'imgWidth'))

# The ratio between the desired image height and width is used to ensure
# that a complete face is captured:
imgHeightWidthRatio = imgHeight / imgWidth
print "imgHeightWidthRatio = " + str(imgHeightWidthRatio)

# Verify that the cascade file exists:
cascPath = './haarcascade_frontalface_default.xml'
if not os.path.isfile(cascPath):
  print 'The face recognition cascade file (' + cascPath + ') does not exist.'
  quit(1)

faceCascade = cv2.CascadeClassifier(cascPath)

# A single parameter should be specified on the command line -- the name of
# the person for whom training images are being captured.  The name should
# not have any spaces:
if len(sys.argv) != 2:
  print 'A single argument should be specified on the command line -- '
  print 'the name of the person for whom training images are being captured.'
  quit(1)

trainingName = sys.argv[1]
print 'Capturing training images for ' + trainingName

# If a subdirectory by that name doesn't already exist, then create it:
folderName = os.path.join('training', trainingName)
if not os.path.isdir(folderName):
  os.makedirs(folderName)
if not os.path.isdir(folderName):
  print 'Unable to create a subdirectory for the training images.'
  quit(1)

# Based on the list of images in the subdirectory (where each file name is
# an integer number), determine what the next file name should be:
fileNum = 1
for fileName in os.listdir(folderName):
  baseName = os.path.basename(fileName)
  if baseName.isdigit() and int(baseName) > fileNum:
    fileNum = int(baseName)
    
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
    # The following calculations are used to increase the size and shape
    # that surrounds the detected face.  Normally, the routine returns a
    # perfect square that chops off the top of the head and the chin.  
    # By increasing the height, we hope to increase the characteristics
    # of the face for recognition puposes:
    faceX1 = x
    faceX2 = x + w
    faceY1 = y - (h * ((imgHeightWidthRatio - 1) / 2))
    faceY2 = y + h + (h * ((imgHeightWidthRatio - 1) / 2))
    
    cv2.rectangle(frame, (faceX1, faceY1), (faceX2, faceY2), (0, 255, 0), 2)
    face_found = True

  cv2.imshow('Video', frame)

  key = cv2.waitKey(1) & 0xFF

  if face_found == True and key == ord('k'):
    for (x, y, w, h) in faces:
      faceX1 = x
      faceX2 = x + w
      faceY1 = y - (h * ((imgHeightWidthRatio - 1) / 2))
      faceY2 = y + h + (h * ((imgHeightWidthRatio - 1) / 2))

      face_img = gray[y:y+h, x:x+w]
      face_img = gray[faceY1:faceY2, faceX1:faceX2]
      face_img = cv2.resize(face_img, (imgWidth,imgHeight), interpolation=cv2.INTER_CUBIC)
      cv2.imwrite(os.path.join(folderName, str(fileNum) + '.png'), face_img)
      fileNum = fileNum + 1
    time.sleep(1)
    face_found = False

  if key == ord('q'):
    break

video_capture.release()
cv2.destroyAllWindows()

