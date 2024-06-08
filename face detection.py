import cv2 as cv

facecascade=cv.CascadeClassifier('haarcascade_frontalface_default.xml')

img= cv.imread('photo.jpg')
imgrsize=cv.resize(img,(700,500))
imggray=cv.cvtColor(imgrsize,cv.COLOR_BGR2GRAY)

faces= facecascade.detectMultiScale(imggray,1.3,5)

for (x,y,w,h) in faces:
    cv.rectangle(imgrsize,(x,y),(x+w,y+h),(255,0,0),2)

cv.imshow('window',imgrsize)
cv.waitKey(0)