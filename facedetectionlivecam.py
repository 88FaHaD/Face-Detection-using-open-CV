import cv2 as cv

facecascade=cv.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("Error: Couldn't open camera.")
    exit()

while True:
    ret, frame = cap.read()
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    faces=facecascade.detectMultiScale(gray_frame,1.3,5)
    for(x,y,w,h) in faces:
        cv.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

    cv.imshow('Video',frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
