import numpy as np
import cv2
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--content-path', type=str, dest='content_path', help='Content image or folder of images')
parser.add_argument('--style-path', type=str, dest='style_path', help='Style image or folder of images')
parser.add_argument('--out-path', type=str, dest='out_path', help='Output folder path')
parser.add_argument('--weight-path', type=str, dest='weight_path', help='weights path')
args = parser.parse_args()



def disp_img(img):
	cv2.imshow('result',img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def kmeans(content_path, style_path, out_path, weight_path, K):

	img = cv2.imread(content_path)
	Z = img.reshape((-1,3))

	# convert to np.float32
	Z = np.float32(Z)

	# define criteria, number of clusters(K) and apply kmeans()
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
	# K = 8
	ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

	# Now convert back into uint8, and make original image
	center = np.uint8(center)
	res = center[label.flatten()]
	res2 = res.reshape((img.shape))
	r, c, p = img.shape
	# seg = np.zeros((r, c, p, K))
	
	# disp_img(res2)
	# print(img.shape)
	# print(res.shape)


	for i in range(center.shape[0]):
		print('mask_', i)
		mask = res2 == center[i]
		masked = np.multiply(mask, img)


		cv2.imwrite(out_path+'/masked_'+str(i+1)+'.png', masked)

		os.system('python style_transfer.py \
			--alpha 0.9 \
			--style-path '+style_path+'/style_'+str(i+1)+'.png \
			--content-path '+out_path +'/masked_'+str(i+1)+'.png \
			--out-path '+out_path+' \
			--live-path 0 \
			--weight-path '+weight_path)

		style = cv2.imread(out_path+'/masked_'+str(i+1)+'_style_'+str(i+1)+'.png')
		
		row, col = mask.shape[0:2]
		style = style[0:row, 0:col, :]
		
		mask_style = np.multiply(style, mask)
		
		if i == 0:
			seg = mask_style
		else:
			seg = cv2.add(seg, mask_style)
		
		disp_img(seg)

	cv2.imwrite('masked_result.png', seg)




kmeans(args.content_path, args.style_path, args.out_path, args.weight_path, 4)

