import cv2

#faceCascade = cv2.CascadeClassifier('haarcascade_eye.xml')
ballCascade = cv2.CascadeClassifier('ball_cascade.xml')
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    eye = ballCascade.detectMultiScale(img, 1.1,4)
    for (x,y,w,h) in eye:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)


    cv2.imshow("Result", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#i = cv2.waitKey(0)
#while(cap.isOpened()):
#	ret, frame = cap.read()
#
#	cv2.imshow('frame', frame)
#	key = cv2.waitKey(1) & 0xFF 
#	if key == ord("q"):
#		break
#
#	cap.release()
#	cv2.destroyAllWindows()
