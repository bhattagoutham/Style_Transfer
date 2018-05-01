import numpy as np
import cv2
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--style-path', type=str, dest='style_path', help='Style image or folder of images')
parser.add_argument('--out-path', type=str, dest='out_path', help='Output folder path')
parser.add_argument('--weight-path', type=str, dest='weight_path', help='weights path')
args = parser.parse_args()

def live_style(style_path, out_path, weight_path):

    cap = cv2.VideoCapture(0)

    while(True):

        print('capturing frame...')
        ret, frame = cap.read()

        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # res = cv2.resize(gray,None,fx=1/2, fy=1/2, interpolation = cv2.INTER_CUBIC)
        cv2.imwrite('frame.jpg', frame)
        # print(gray.shape)
        os.system('python style_transfer.py \
    			--alpha 0.9 \
    			--style-path '+style_path+' \
    			--content-path frame.jpg \
    			--out-path '+out_path+' \
    			--live-path 1 \
                --weight-path '+weight_path)


        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

live_style(args.style_path, args.out_path, args.weight_path)
