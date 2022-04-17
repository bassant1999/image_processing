#steps:
#get rid of noise s sun light by transforming to color space as HSL so change color space of image to HSL space.
#get the two lines (through which car will move) --> we may use canny edge detection to get the edges but we won't use it now but we will use Image Thresholding
#use Hough Line Transformation.
#then color the regoin between the formed two lines easily
import os
import re
import cv2
import numpy as np
import tqdm
import matplotlib.pyplot as plt
# get file names of frames
col_frames = os.listdir('frames/')
col_frames.sort(key=lambda f: int(re.sub('\D', '', f)))

# load frames
col_images=[]
for i in tqdm.tqdm(col_frames):
    img = cv2.imread('frames/'+i)
    col_images.append(img)
    
#frame index
index = 100
# plot
plt.figure(figsize=(10,10))
img = col_images[index]
image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(image)
plt.show()

#crop the part of the image that i care about it
copy = np.zeros_like(col_images[idx][:,:,0])
polygon = np.array([[50,270], [220,160], [360,160], [480,270]])
cv2.fillConvexPoly(copy, polygon, 1)

# apply polygon on the image
img = cv2.bitwise_and(col_images[index][:,:,0], col_images[index][:,:,0], mask=stencil)
plt.figure(figsize=(10,10))
plt.imshow(img, cmap= "gray")
plt.show()

#image thresholding
ret, thresh = cv2.threshold(img, 130, 145, cv2.THRESH_BINARY)

#houghline transform
lines = cv2.HoughLinesP(thresh, 1, np.pi/180, 30, maxLineGap=200)
img_copy = col_images[index].copy()
i = 0
#Hough lines
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(img_copy, (x1, y1), (x2, y2), (0,128,0), 3)  # cv2.line(): https://www.geeksforgeeks.org/python-opencv-cv2-line-method/
# plot frame
plt.figure(figsize=(10,10))
fixed_img = cv2.cvtColor(dmy, cv2.COLOR_BGR2RGB)
plt.imshow(fixed_img)
plt.show()

#color between the line
#use fillConvexPoly()



#source: https://www.analyticsvidhya.com/blog/2020/05/tutorial-real-time-lane-detection-opencv/  and https://www.kdnuggets.com/2017/07/road-lane-line-detection-using-computer-vision-models.html
