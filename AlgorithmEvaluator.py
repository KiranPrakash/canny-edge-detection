from PIL import Image
import numpy as num, math
import numpy as np 
from math import pi, sqrt, exp
from copy import deepcopy
import matplotlib.pyplot as plt
from numpy import matrix
import matplotlib.cm as cm
import random

#Step2: Creating a 1D Gaussian Mask G to convolve with I. Standard Deviation for the function "gauss( , )"
# n/2 represents the number of points on either side of 0 on number line
def gauss(n,sigma):
    r = range(-int(n/2),int(n/2)+1)
    return [1 / (sigma * sqrt(2*pi)) * exp(-float(x)**2/(2*sigma**2)) for x in r]
g = gauss(5,2)
#1D Mask for the first Derivative of the Gaussian Kernel
def gaussDeriv(n,sigma):
    r = range(-int(n/2),int(n/2)+1)
    return [-x / (sigma**3*sqrt(2*pi)) * exp(-float(x)**2/(2*sigma**2)) for x in r]

# Canny Edge detector   
def Canny(In, Out):

#Step2: Creating a 1D Gaussian Mask G to convolve with I. Standard Deviation for the function "gauss( , )"
# n/2 represents the number of points on either side of 0 on number line    
    Igx= []
    for i in range(len(In[:,0])):
        x = np.convolve(In[i,:], g)
        Igx.append(x)
    Igx = np.array(np.matrix(Igx))        


#Step4 : 1D Mask for the first Derivative of the Gaussian in y direction using np.convolve function
#Transpose is taken in order to do the convultion downwards along the y axis
    IgyT=[]
    for i in range (len(In[0,:])):
        y = np.convolve(In[:,i], g)
        IgyT.append(y) 
    Igy = np.transpose(IgyT) 

#Step5: x component convolved with the derivative of the Gaussian, we get the Idgx
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

#Step 6: Calculation for magnitude of the edge response by taking square root of sum of squares of the x and  y convolved in step 4
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

#Step7: Non-Maximum Suppression Algorithm

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
    highT = 3 # The Non Maximum suppression matrix was checked for several points of 
    lowT = 1  #thresholds to suppress the non edge points
    maxm = 255 # we would be using this to set the pixel in order to make it a edge in the following for loops
        
    for i in range(len(Hyst[:,0])-1):    
        for j in range(len(Hyst[0,:])-1):
	   u = i
	   v = j
	   while((u!=0)&(v!=0)):
	       if (Hyst[u,v] >=highT):
		  Hyst[u,v] = maxm
		  try:
		      if (lowT<=Hyst[u+1,v]<highT):
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

  ## Finding the  Metrics for the Edge detector for the given set of images

    Image = np.array(Out)
 #intializing the counter
    TP=TN=FP=FN=0.0 
    s =np.shape(Image)
    p = s[0]
    q = s[1]

    for i in range(p):    
        for j in range(q):
           if(( Hyst[i,j] == 255 and Image[i,j] > 0)):
               TP+=1
           elif (( Hyst[i,j] == 0  and Image[i,j] == 0)):
               TN+=1
           elif (( Hyst[i,j] == 255 and Image[i,j] == 0)):
               FP+=1 
           else:
               FN+=1
               


    Sensitivity                     = TP/(TP+FN) #Calculating evaluation parameters from the iven formulae
    Specificity                     = TN/(TN+FP)
    Precision                       = TP/(TP+FP)
    Negative_Prediction_Value       = TN/(TN+FN)
    Fall_out                        = FP/(FP+TN)
    False_Negative_Rate             = FN/(FN+TP)
    #False_Discovery_Rate            = FP/(FP+TP)
    Accuracy                        = (TP+TN)/(TP+FN+TN+FP)
    F_score                         = 2 * TP/((2*TP)+FP+FN)
    #Matthew_Correlation_Coefficient = ((TP*TN) - (FP*FN))/(sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)))
               
           
    print 'RESUTLTS FOR IMAGES after comparision '
   #Printing the results
    print 'Sensitivity is',Sensitivity    
    print 'Specificity is',Specificity
    #print 'Precision is',Precision
    print 'Negative Prediction Value is',Negative_Prediction_Value
    print 'Fall_out is',Fall_out
    print 'False_Negative_Rate is',False_Negative_Rate
    #print 'False Discovery Rate is',False_Discovery_Rate
    print 'Accuracy is',Accuracy
    print 'F_score is',F_score
   # print 'MCC is',Matthew_Correlation_Coefficient

   # respective plots of the input and output images   
    plt.figure()
    plt.subplot(2,1,1)
    plt.title('Input Image')
    plt.imshow(In,cmap = cm.gray)
    plt.subplot(2,1,2)
    plt.title('Output Image')
    plt.imshow(Hyst,cmap = cm.gray)
    
    plt.show()

#PLEASE CHANGE THE IMAGE PATH

#Reading the Images and converting to grayscale array which is the input to the canny function
In1= Image.open('C:/Users/Kiran/Desktop/Documents/UCF/Academics/Computer Vision/Assignment1/Prob2/input_image1.jpg').convert('L')
In1 = num.array(In1)
Out1=Image.open('C:/Users/Kiran/Desktop/Documents/UCF/Academics/Computer Vision/Assignment1/Prob2/output_image1.png').convert('L')
Out1= num.array(Out1)
Canny(In1, Out1)# Canny edge detector function call

In2= Image.open('C:/Users/Kiran/Desktop/Documents/UCF/Academics/Computer Vision/Assignment1/Prob2/input_image2.jpg').convert('L')   
In2 = num.array(In2)
Out2= Image.open('C:/Users/Kiran/Desktop/Documents/UCF/Academics/Computer Vision/Assignment1/Prob2/output_image2.png').convert('L')
Out2= num.array(Out2)
Canny(In2, Out2)# canny edge detector function call for the second image

def GaussNoise(I): # Gaussian noise function
    row,col = np.shape(I)
    mean = 0
    var = 100
    sig = math.sqrt(var)
    gauss = np.random.normal(mean,sig,(row,col)) #generating random samples from Gaussian distribution
    gauss = gauss.reshape(row,col) # Image is reshaped 
    NoisyI= I + gauss
    return NoisyI


def saltPepper(I,P):
    O = np.zeros(I.shape,np.uint8) # Output
    T=1- P# threshold is the 1 minus the sd
    
    for i in range(len(I[:,0])):
        for j in range(len(I[0:,])):
            R = random.random()
            if R < P: 
                O[i,j] = 0 #setting black dots
            elif R > T:
                O[i,j] = 255 #setting white dots
            else:
                O[i,j] = I[i,j]
    return O
    
# Gaussian Noise addition to input images
NoiseImage= GaussNoise(In1)
NoiseImage= saltPepper(NoiseImage, 0.25)
Canny(NoiseImage, Out1)







