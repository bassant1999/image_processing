#steps:
#get rid of noise s sun light by transforming to color space as HSL so change color space of image to HSL space.
#get the two lines (through which car will move) --> we will use hough algorithm but we will use Image Thresholding
#use Hough Line Transformation.
#then color the regoin between the formed two lines easily
import os
import re
import cv2
import numpy as np
import tqdm
import matplotlib.pyplot as plt

# load frames
import cv2
import numpy as np
import sys
argv = sys.argv
print(argv[1])

col_images=[]

# Create a VideoCapture object and read from input file
cap = cv2.VideoCapture('project_video.mp4')
print("hi")

# Check if camera opened successfully
if (cap.isOpened()== False):
    print("Error opening video file")

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

# Read until video is completed
while(cap.isOpened()):
	
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:
        col_images.append(frame)
        # Write the frame into the file 'output.avi'
        #out.write(frame)


        # Display the resulting frame
        cv2.imshow('Frame', frame)

        # Press Q on keyboard to exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Break the loop
    else:
        break

# When everything done, release
# the video capture object
cap.release()
out.release()


# Closes all the frames
cv2.destroyAllWindows()




    
#frame index
index = 10
# plot
plt.figure(figsize=(10,10))
img = col_images[index]
image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(image)
plt.show()

#crop the part of the image that i care about it
copy = np.zeros_like(col_images[idx][:,:,0])
polygon = np.array([[350,600], [550,450], [700,450], [1100,600]])
cv2.fillConvexPoly(copy, polygon, 1)

# apply polygon on the image
img = cv2.bitwise_and(col_images[index][:,:,0], col_images[index][:,:,0], mask=copy)
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


#color the lanes with green
lines = cv2.HoughLinesP(thresh, 1, np.pi/180, 30, maxLineGap=200)
# create a copy of the original frame
dmy = col_images[idx].copy()
i = 0
# draw Hough lines
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(dmy, (x1, y1), (x2, y2), (0,128,0), 3)  # cv2.line(): https://www.geeksforgeeks.org/python-opencv-cv2-line-method/
# plot frame
plt.figure(figsize=(10,10))
fixed_lena = cv2.cvtColor(dmy, cv2.COLOR_BGR2RGB)
plt.imshow(fixed_lena)
plt.show()


#color between the lanes with green
#we want large_y with small_x (as (89,269), large_y with large_x (as (406,166)), small_y as (257,166) and (331,175)
import math

max_y = 0
max_x = 0
min_y = math.inf
min_x = math.inf
list_x = []
for line in lines:
    x1, y1, x2, y2 = line[0]
    print ("(",x1,y1,")","(",x2,y2,")")
    #get maximum y
    if y1 > max_y:
        max_y = y1
    if y2 > max_y:
        max_y = y2
    #get minimum y
    if y1 < min_y:
        min_y = y1
    if y2 < min_y:
        min_y = y2
    #get maximum x
    if x1 > max_x:
        max_x = x1
    if x2 > max_x:
        max_x = x2
    #get manimum x
    if x1 < min_x:
        min_x = x1
    if x2 < min_x:
        min_x = x2
    list_x.append(x1)
    list_x.append(x2)

list_xs = []
for line in lines:
    x1, y1, x2, y2 = line[0]
    if y1 == min_y:
        list_xs.append(x1)
    if y2 == min_y:
        list_xs.append(x2)
        
print(max_y, min_x,max_x,min_y)
print("list",list_xs)
list_xs.sort()
if(len(list_xs) > 1):
    print("here again")
    first_x = list_xs[0]
    second_x = list_xs[len(list_xs)-1]
else:
    mid = int((max_x + min_x)/2)
    mid_max = int((max_x - mid)/2)
    second_x = max_x - mid_max
    mid_min = int((mid - min_x)/2)
    first_x = min_x +mid_min
    print("h",abs(min_x - list_xs[0]) ,abs(max_x - list_xs[0]))
    if(abs(min_x - list_xs[0]) < abs(max_x - list_xs[0])):
        
        first_x = int((list_xs[0] + first_x)/2)
        s_x = (list_xs[0] + max_x)/2
        second_x = int((s_x + second_x)/2)
    else:
        print("here")
        second_x = int((list_xs[0] + second_x)/2)
        f_x = (list_xs[0] + min_x)/2
        first_x = int((f_x + first_x)/2) 

        
    first_x = min_x+150
    second_x = max_x-80
# list_x.sort()
# print(list_x)
# print(len(list_x))
# length = len(list_x)
# if(length % 2 ==0):
#     first_x = list_x[int((length/2)) - 1]
#     second_x = list_x[int(length/2) ]
# else:
#     first_x = list_x[int((length/2))]
#     second_x = list_x[int(length/2) + 1]
# print(first_x , second_x)
# print(int(5/2))
print(first_x,second_x)
#use fillConvexPoly()
from PIL import Image
from numpy import array
my = col_images[idx].copy()
imag1 = array(my)
# poly1 = np.array([[79,269], [257,166], [331,166], [435,269]])
print(min_x-20,max_y, first_x,min_y, second_x, min_y, max_x+20 , max_y)
poly1 = np.array([[min_x-20,max_y], [first_x,min_y], [second_x,min_y], [max_x+20,max_y]])
cv2.fillConvexPoly(imag1, poly1, (0,128,0))
# plot polygon
plt.figure(figsize=(10,10))
plt.imshow(imag1, cmap= "gray")
plt.show()

#another solution to color between the lanes but very slow
list_lines =[]
for line in lines:
    x1, y1, x2, y2 = line[0]
    list_lines.append([x1,y1])
    list_lines.append([x2,y2])
print(list_lines)
my_img = col_images[idx].copy()
image_T = array(my_img )
# poly1 = np.array([[79,269], [257,166], [331,166], [435,269]])
for line in list_lines:
    for line1 in list_lines:
        for line2 in list_lines:
            for line3 in list_lines:
                print(line, line1,line2,line3)
                poly_T = np.array([line, line1,line2,line3])
                cv2.fillConvexPoly(image_T, poly_T, (0,128,0))
                # plot polygon
plt.figure(figsize=(10,10))
plt.imshow(image_T, cmap= "gray")
plt.show()




#source: https://www.analyticsvidhya.com/blog/2020/05/tutorial-real-time-lane-detection-opencv/  and https://www.kdnuggets.com/2017/07/road-lane-line-detection-using-computer-vision-models.html
