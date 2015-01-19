#!/usr/bin/python
# ==============================================================================
# Name: capture_training_images.py
# Desc: This program is used to capture training images that will be used by
#   the face recognition program.  
# Change History:
# 2015-02-01 JSP: Created.
# ==============================================================================

import cv2
import sys
import time

# Verify that the cascade 
cascPath = sys.argv[1]
faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0)

while True:
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

  # Put a blue box around all the faces that were found in the image:
  for (x, y, w, h) in faces:
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    face_found = True

  cv2.imshow('Video', frame)

  key = cv2.waitKey(1) & 0xFF

  if face_found == True and key == ord('k'):
    print "Keeping face"
    for (x, y, w, h) in faces:
      print "X:" + str(x) + ", Y:" + str(y) + ", W:" + str(w) + ", H:" + str(h)
      cv2.imwrite('face.png', frame[y:y+h, x:x+w])
    time.sleep(1)
    face_found = False

  if key == ord('q'):
    break

video_capture.release()
cv2.destroyAllWindows()

