#!/usr/bin/python
# ==============================================================================
# Name: capture_training_images.py
# Desc: This program is used to capture training images that will be used by
#   the face recognition program.  
# Change History:
# 2015-02-01 JSP: Created.
# ==============================================================================

import cv2
import os.path
import sys
import time

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
if not os.path.isdir(trainingName):
  os.makedirs(trainingName)
if not os.path.isdir(trainingName):
  print 'Unable to create a subdirectory for the training images.'
  quit(1)

# Based on the list of images in the subdirectory (where each file name is
# an integer number), determine what the next file name should be:
fileNum = 1
for fileName in os.listdir(trainingName):
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
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    face_found = True

  cv2.imshow('Video', frame)

  key = cv2.waitKey(1) & 0xFF

  if face_found == True and key == ord('k'):
    for (x, y, w, h) in faces:
      face_img = gray[y:y+h, x:x+w]
      face_img = cv2.resize(face_img, (100,100), interpolation=cv2.INTER_CUBIC)
      cv2.imwrite(os.path.join(trainingName, str(fileNum) + '.png'), face_img)
      fileNum = fileNum + 1
    time.sleep(1)
    face_found = False

  if key == ord('q'):
    break

video_capture.release()
cv2.destroyAllWindows()

