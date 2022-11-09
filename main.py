import cv2 as cv

from random import randrange
trained_face_data = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

#img = cv.imread('apj.png')
#img = cv.imread

webcam = cv.VideoCapture(0)

while True:
    succesfull_frame_read, frame = webcam.read()
    grey_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    face_coordinates = trained_face_data.detectMultiScale(grey_img)
    for (x, y, w, h) in face_coordinates:
        cv.rectangle(frame, (x, y), (x+w, y+h), (randrange(256), randrange(256), randrange(256)), 2)
    cv.imshow('Clever Programmer Face detector', frame)
    c = cv.waitKey(1)
    if c == 27:
        break
webcam.release()
cv.destroyAllWindows()

"""

for (x, y, w, h) in face_coordinates:
    cv.rectangle(img, (x, y), (x+w, y+h), (randrange(256), randrange(256), randrange(256)), 2)
#print(face_coordinates)

cv.imshow('Clever Programmer Face detector', img)

cv.waitKey()



print("Code Completed")"""

