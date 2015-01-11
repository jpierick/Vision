import numpy as np
import cv2
import cv

cap = cv2.VideoCapture(0)

while(True):
  ret, frame = cap.read()
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  cv2.imshow('frame', gray)

#  res = cv2.resize(gray, None, fx=2, fy=2, interpolation = cv2.INTER_CUBIC)
  res = cv2.resize(gray, (92,112), interpolation = cv2.INTER_CUBIC)
  cv2.imshow('resize', res)

  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

cap.release()
cv2.destroyAllWindows()
