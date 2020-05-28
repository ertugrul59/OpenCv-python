import cv2
#https://github.com/opencv/opencv/tree/master/data/haarcascades
#https://gist.githubusercontent.com/pknowledge/b8ba734ae4812d78bba78c0a011f0d46/raw/aa2d79af55c9bf9981ec887cf09d866a2742c148/face_detection.py
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') # pretrained model
# Read the input image
#img = cv2.imread('test.png')
#cap = cv2.VideoCapture('test.mp4')
cap = cv2.VideoCapture(0)
while cap.isOpened():
    _, img = cap.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y , w ,h) in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0 , 0), 3)

    # Display the output
    cv2.imshow('img', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()