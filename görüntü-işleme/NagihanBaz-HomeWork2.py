# 171805024-Nagihan Baz
"""
Created on Thu Nov 17 18:36:23 2022

@author: USER
"""

import matplotlib.pyplot as plt
#from skimage import exposure
from skimage.exposure import match_histograms
import cv2
import numpy as np

#???img_path =imgr'C:\Users\USER\Desktop\res/'+str(j)+'_'+str(i)+'.jpg' 

# read a image using imread
img = cv2.imread(r'C:\Users\USER\Desktop\res\kedi.jpg', 0)

# of a image using cv2.equalizeHist()
equ = cv2.equalizeHist(img)

# stacking images side-by-side
res = np.hstack((img, equ))

# show image input vs output
cv2.imshow('equalization', res)

# Specification
# reading main image again because code didn't work
img1 = cv2.imread(r'C:\Users\USER\Desktop\res\kedi.jpg')

# checking the number of channels
print('No of Channel is: ' + str(img1.ndim))

# reading reference gray image
img2 = cv2.imread(r'C:\Users\USER\Desktop\res\gray-bg.jpg')

# checking the number of channels 2
print('No of Channel is: ' + str(img2.ndim))

image = img1
reference = img2

matched = match_histograms(image, reference ,
						multichannel=True)


fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3,
									figsize=(8, 3),
									sharex=True, sharey=True)

for aa in (ax1, ax2, ax3):
	aa.set_axis_off()

ax1.imshow(image)
ax1.set_title('Source')
ax2.imshow(reference)
ax2.set_title('Reference')
ax3.imshow(matched)
ax3.set_title('Matched')

plt.tight_layout()
plt.show()

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(5,5))

# find histogram of an image
plt.hist(img.ravel(),300,[0,300])
plt.show()


plt.tight_layout()
plt.show()