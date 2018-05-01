import numpy as np
import cv2
import os

def disp_img(img):
	cv2.imshow('res2',img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def kmeans(file_path, K):

	img = cv2.imread(file_path)
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
		cv2.imwrite('/home/goutham/WCT-TF-master/samples/masked_'+str(i+1)+'.png', masked)
		os.system('python style_transfer.py \
			--alpha 0.9 \
			--style-path /home/goutham/WCT-TF-master/samples/style_'+str(i+1)+'.png \
			--content-path /home/goutham/WCT-TF-master/samples/masked_'+str(i+1)+'.png \
			--out-path /home/goutham/test')

		style = cv2.imread('/home/goutham/test/masked_'+str(i+1)+'_style_'+str(i+1)+'.png')
		
		row, col = mask.shape[0:2]
		style = style[0:row, 0:col, :]
		
		mask_style = np.multiply(style, mask)
		
		if i == 0:
			seg = mask_style
		else:
			seg = cv2.add(seg, mask_style)
		
		disp_img(seg)

	cv2.imwrite('/home/goutham/test/result.png', seg)




kmeans('/home/goutham/WCT-TF-master/samples/21.JPG', 4)
