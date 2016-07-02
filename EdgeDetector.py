from PIL import Image
import numpy as num, math
import numpy as np 
from math import pi, sqrt, exp
from copy import deepcopy
import matplotlib.pyplot as plt
from numpy import matrix
import matplotlib.cm as cm


#Step1: Creating a 1D Gaussian Mask G to convolve with I. Standard Deviation for the function "gauss( , )"
# n/2 represents the number of points on either side of 0 on number line
def gauss(n,sigma):
    r = range(-int(n/2),int(n/2)+1)
    return [1 / (sigma * sqrt(2*pi)) * exp(-float(x)**2/(2*sigma**2)) for x in r]

#1D Mask for the first Derivative of the Gaussian Kernel
def gaussDeriv(n,sigma):
    r = range(-int(n/2),int(n/2)+1)
    return [-x / (sigma**3*sqrt(2*pi)) * exp(-float(x)**2/(2*sigma**2)) for x in r]

# Canny Edge detector   
def Canny(I):

#Step2: Creating a 1D Gaussian Mask G to convolve with I. Standard Deviation for the function "gauss( , )"
# n/2 represents the number of points on either side of 0 on number line    
    Igx= []
    g = gauss(7,2)
    for i in range(len(I[:,0])):
        x = np.convolve(I[i,:], g)
        Igx.append(x)
    Igx = np.array(np.matrix(Igx))# Ix'


#Step3 : 1D Mask for the first Derivative of the Gaussian in y direction using np.convolve function
#Transpose is taken in order to do the convultion downwards along the y axis
    IgyT=[]
    for i in range (len(I[0,:])):
        y = np.convolve(I[:,i], g)
        IgyT.append(y) 
    Igy = np.transpose(IgyT)

#Step4: x component convolved with the derivative of the Gaussian, we get the Idgx
    gd = gaussDeriv(7,2)
    Igdx=[] 
    for i in range (len(Igx[:,0])):
        x = np.convolve(Igx[i,:], gd)
        Igdx.append(x)

# y component convolved with the derivatiev of the Gaussian, we get Idgy
    IgdyT=[]
    for i in range (len(Igy[0,:])):
        y = np.convolve(Igy[:,i], gd)
        IgdyT.append(y) 

    Igdy= np.transpose(IgdyT)
    
#Step 5: Calculation for magnitude of the edge response by taking square root of sum of squares of the x and  y convolved in step 4
    Igdxsq= np.square(Igdx)
    Igdysq= np.square(Igdy)

    Mxy =[]
    for i in  range (len(Igdxsq)):
        temp = []
        for j in range (len(Igdysq[0,:])):
            temp.append(sqrt(Igdxsq[i,j] + Igdysq[i,j]))
            if(j == len(Igdysq[0,:])-1):
                Mxy.append(temp)
    Mxy = np.array(np.matrix(Mxy)) 

#Step6: Non-Maximum Suppression Algorithm

    A= np.array(np.matrix(Igdx))
    B= np.array(np.matrix(Igdy))
    AngleDeg =[]
    
#  finding the Angle Theta using atan2 for Gaussian derivatives Ix and Iy
    for i in range(len(Igdx)):
        temp=[]
        for j in range(len(Igdy[0,:])):
            temp.append((math.atan2(B[i,j],A[i,j]))*180/pi)
            if(j == len(Igdy[0,:])-1):
                AngleDeg.append(temp)

# converting the angles from radians to Degrees and store it in an array matrix       
    Angle= np.array(np.matrix(AngleDeg))

    MxyTemp = Mxy 
    NonMax  = deepcopy(Mxy)


    for i in range(len(Angle[:,0])):
        for j in range(len(Angle[0,:])):
            try:
            #Horizontal Edge
            
                if ((-22.5< Angle[i,j] <= 22.5) | ( -157.5 < Angle[i,j] <= 157.5)):
                    if((MxyTemp[i,j] < MxyTemp[i+1,j]) | (MxyTemp[i, j] < MxyTemp[i-1,j])):
                        NonMax[i,j] = 0
                
            #Vertical Edge
            
                if ((-112.5 < Angle[i,j] <= -67.5) | ( 67.5 < Angle[i,j] <= 112.5)):
                    if((MxyTemp[i,j] < MxyTemp[i,j+1]) | (MxyTemp[i, j] < MxyTemp[i,j-1])):
                        NonMax[i,j] = 0
                    
                    
            #+45 Degree Edge
            
                if ((-67.5 < Angle[i,j] <= -22.5) | ( 112.5 < Angle[i,j] <= 157.5)):
                    if((MxyTemp[i,j] < MxyTemp[i+1,j+1]) | (MxyTemp[i, j] < MxyTemp[i+1,j+1])):
                        NonMax[i,j] = 0
                    
            #-45 degree Edge
            
                if ((-157.5 < Angle[i,j] <= -112.5) | (22.5 < Angle[i,j] <= 67.5 )):
                    if((MxyTemp[i,j] < MxyTemp[i-1,j-1]) | (MxyTemp[i, j] < MxyTemp[i+1,j+1])):
                        NonMax[i,j] = 0
            

            except IndexError:
                pass


#Hysterisis Thresholding
# We would be suppressing the residuals in 8 directions around the pixel

    NonMax =(matrix(NonMax))
    Hyst = deepcopy(NonMax)
    u = v = 0
    highT = 4.5 # The Non Maximum suppression matrix was checked for several points of 
    lowT = 1.5  #thresholds to suppress the non edge points
    maxm = 255 # we would be using this to set the pixel in order to make it a edge in the following for loops
        
    for i in range(len(Hyst[:,0])-1):    
        for j in range(len(Hyst[0,:])-1):
	   u = i
	   v = j
	   while((u!=0)&(v!=0)):
	       if (Hyst[u,v] >=highT):
		  Hyst[u,v] = maxm
		  try:
		      if (lowT<=Hyst[u+1,v] < highT):
		          Hyst[u+1,v] = maxm
			  u = u+1
			  v = v			
		      elif (lowT<=Hyst[u-1,v]<highT):
			  Hyst[u-1,v] = maxm
			  u = u-1 
			  v= v
		      elif (lowT<=Hyst[u+1,v+1]<highT):
			  Hyst[u+1,y+1] = maxm
			  u = u+1
			  v = v+1
		      elif (lowT<=Hyst[u-1,v-1]<highT):
			  Hyst[u-1,v-1] = maxm
			  u = u-1
			  v = v-1
		      elif (lowT<=Hyst[u,v+1]<highT):
			  Hyst[u,v+1] = maxm
			  u = u
			  v = v+1
		      elif (lowT<=Hyst[u,v-1]<highT):
			  Hyst[u,v-1] = maxm
			  u = u
			  v = v-1
		      elif (lowT<=Hyst[u-1,v+1]<highT):
			  Hyst[u-1,v+1] = maxm
			  u = u-1
			  v = v+1
		      elif (lowT<=Hyst[u+1,v-1]<highT):
			  Hyst[u+1,v-1] = maxm
			  u = u+1
			  v = v-1
		      else: 
			  u = 0
			  v = 0
                  except IndexError: 
                                    u = 0
                                    v = 0
	       elif (lowT<= Hyst[u,v]<highT):
	           Hyst[u,v] = maxm
	       else:
	           Hyst[u,v] = 0 
                   u = 0
                   v = 0  
    
    plt.figure()
    
    plt.subplot(3,3,1)
    plt.title('Input Image')
    plt.imshow(I,cmap = cm.gray)
    
    plt.subplot(3,3,2)
    plt.title('Image masked with Gaussian along x axis: Ix')
    plt.imshow(Igx,cmap = cm.gray)
    
    
    plt.subplot(3,3,3)
    plt.title('Image masked with Gaussian along y axis : Iy')
    plt.imshow(Igy,cmap = cm.gray)
    
    
    plt.subplot(3,3,4)
    plt.title('Image masked with Gaussian Derivative along x axis : Ix`')
    plt.imshow(Igdx,cmap = cm.gray)
    
    
    plt.subplot(3,3,5)
    plt.title('Image masked with Gaussian Derivative along y axis : Iy`')
    plt.imshow(Igdy,cmap = cm.gray)
    
    
    plt.subplot(3,3,6)
    plt.title('Image Edge Response Magnitude M(x,y) ')
    plt.imshow(Mxy,cmap = cm.gray)
    
    
    plt.subplot(3,3,7)
    plt.title('Non Maximum Suppressed')
    plt.imshow(NonMax,cmap = cm.gray)
    
    
    plt.subplot(3,3,8)
    plt.title('After Hysterisis Thresholding')
    plt.imshow(Hyst,cmap = cm.gray)
    
    plt.show() 


#Please change the path to image folder

Image1= num.array(Image.open('C:/Users/Kiran/Desktop/Documents/UCF/Academics/Computer Vision/Assignment1/Prob1/42049.jpg'))   
#Function Called 
Canny(Image1)
