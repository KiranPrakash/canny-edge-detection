from PIL import Image
import numpy as np
import numpy as num
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm

#Please change tthe path
I = num.array(Image.open('C:/Users/Kiran/Desktop/Documents/UCF/Academics/Computer Vision/Assignment1/Berkley Dataset/157055.jpg'))
#Initialising box matrix
matrix = [[1.0/9.0, 1.0/9.0, 1.0/9.0], [1.0/9.0,1.0/9.0,1.0/9.0], [1.0/9.0,1.0/9.0,1.0/9.0]] 
matrix= np.array(matrix)

Shape = np.shape(I) 
I=np.array(I)
In = []

#Convolution of the Box Matrix 30 times
In = cv2.filter2D(I,-1,matrix)
Iresid = []
for i in range(30):
    I= cv2.filter2D(In,-1,matrix)

#Convoluting the image with Gaussian matrices of size (2k+1) 

IG3 = cv2.blur(I,(3,3))
IresidG3 = []
for i in range(30):
    IG3 = cv2.blur(IG3,(3,3))
    IresidG3.append(I - IG3)
IG5 = cv2.blur(I,(5,5))
for i in range(30):
    IG5 = cv2.blur(IG5,(5,5))
IresidG5 = I - IG5
IG7 = cv2.blur(I,(7,7))
for i in range(30):
    IG7 = cv2.blur(IG7,(7,7))
IresidG7 = I - IG7
IG9 = cv2.blur(I,(9,9))
for i in range(30):
    IG9 = cv2.blur(IG9,(9,9))
IresidG9 = I - IG9
IG11 = cv2.blur(I,(11,11))
for i in range(30):
    I_G11 = cv2.blur(IG11,(11,11))
IresidG11 = I - IG11
IG13 = cv2.blur(I,(13,13))
for i in range(30):
    IG13 = cv2.blur(IG13,(13,13))
IresidG13 = I - IG13
IG15 = cv2.blur(I,(15,15))
for i in range(30):
    IG15 = cv2.blur(IG15,(15,15))
IresidG15 = I - IG15
IG17 = cv2.blur(I,(17,17))
for i in range(30):
    IG17 = cv2.blur(IG15,(17,17))
IresidG17 = I - IG17
IG21 = cv2.blur(I,(21,21))
for i in range(30):
    IG21 = cv2.blur(IG21,(21,21))
IresidG21 = I - IG21
IG23 = cv2.blur(I,(23,23))
for i in range(30):
    IG23 = cv2.blur(IG5,(23,23))
IresidG23 = I - IG23
IG25 = cv2.blur(I,(25,25))
for i in range(30):
    IG25 = cv2.blur(IG25,(25,25))
IresidG25 = I - IG25
IG27 = cv2.blur(I,(27,27))
for i in range(30):
    IG27 = cv2.blur(IG27,(27,27))
IresidG27 = I - IG27
IG29 = cv2.blur(I,(29,29))
for i in range(30):
    IG29 = cv2.blur(IG29,(29,29))
IresidG29 = I - IG29
IG31 = cv2.blur(I,(31,31))
for i in range(30):
    IG31 = cv2.blur(IG31,(31,31))
IresidG31 = I - IG31
print Iresid 
print IresidG3

#Smoothing of the image by iterating the box matrix 30 times for the corresponding residual values
#Respective plots
plt.close('all')
plt.figure()
plt.imshow(In, cmap = cm.gray)
plt.show()
plt.figure()
plt.imshow(IG3, cmap = cm.gray)
plt.show()
