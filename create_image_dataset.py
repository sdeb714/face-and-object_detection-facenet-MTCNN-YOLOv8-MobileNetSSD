import cv2
import time
cpt = 0
maxFrames = 30 # if you want 30 frames only.

cap=cv2.VideoCapture(0)

while cpt < maxFrames:
    ret, frame = cap.read()
    frame=cv2.resize(frame,(640,480))
    cv2.imshow("test window", frame) # show image in window
    cv2.imwrite(r"F:\pythonProject\Pen_dataset\Mobile_%d.jpg" %cpt, frame)
    time.sleep(0.5)
    cpt += 1
    if cv2.waitKey(1)&0xFF==27:
        break
cap.release()
cv2.destroyAllWindows()