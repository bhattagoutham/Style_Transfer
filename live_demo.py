import numpy as np
import cv2
import os

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    res = cv2.resize(gray,None,fx=1/2, fy=1/2, interpolation = cv2.INTER_CUBIC)
    cv2.imwrite('frame.jpg', res)
    # print(gray.shape)
    os.system('python test.py \
			--alpha 0.9 \
			--style-path 2.jpg \
			--content-path frame.jpg \
			--out-path /home/goutham/test \
			--live-path 1')


    # Display the resulting frame
    cv2.imshow('frame',res)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
