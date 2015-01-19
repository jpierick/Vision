import cv2
import sys
import time

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

  for (x, y, w, h) in faces:
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    face_found = True

  key = cv2.waitKey(1) & 0xFF

  cv2.imshow('Video', frame)

  if face_found == True and key == ord('k'):
    cv2.putText(frame, "Keep", (50,50), cv2.FONT_HERSHEY_PLAIN, 3.0, (0,0,255))    
    time.sleep(5)
    face_found = False

  cv2.imshow('Video', frame)

  if key == ord('q'):
    break

video_capture.release()
cv2.destroyAllWindows()

