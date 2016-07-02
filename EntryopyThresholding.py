from PIL import Image
import numpy as num, math
import numpy as np 
from copy import deepcopy
import matplotlib.pyplot as plt
import matplotlib.cm as cm

#I=num.array(Image.open('C:/Users/Kiran/Desktop/Documents/UCF/Academics/Computer Vision/Assignment1/Berkley Dataset/147091.jpg'))


def Thresholding(I):
    
# generate a histogram array and store it in histo
    histo = np.zeros(256)
    for i in range(len(I[:,0])):    
        for j in range(len(I[0,:])):
	   x = I[i,j]
	   histo[x] += 1

# to find the total number of pixels in the  Histogram
    sum_A = []
    total = 0
    for i in range (255):
        total += histo[i]
        sum_A.append(total)
    
# Pixels in the Image Region
    total_pixels = len(I[:,0])*len(I[0,:])
    sum_B = []
    for i in range(255):
        sum_B.append(total_pixels - sum_A[i])
    


#finding the probability distribution A and B region
    ProbA=[]
    for i in range(255):
        if sum_A[i]!=0:
            ProbA.append(histo[i] / sum_A[i])
        else:
            ProbA.append(0)
    
   
    ProbB=[]
    for i in range(255):
        ProbB.append(histo[i]/ sum_B[i])
   
#Finding the entropy of A above the threshold value and B below the threshold value

    EntropyA = []
    sumEntropyA = 0
    for i in range(255):
        try:
            sumEntropyA += (ProbA[i]) * math.log(ProbA[i])
            EntropyA.append(-sumEntropyA)
       
        except ValueError:
            EntropyA.append(0)                

    EntropyB= []
    sumEntropyB=0
    for i in range(255):
        try:
            sumEntropyB += (ProbB[i]) * math.log(ProbB[i])  
            EntropyB.append(sumEntropyB)
        except ValueError:
            EntropyB.append(0)                  


#Total Entropy of A and B combined
    Entropy=[]
    for i in range(255):
        Entropy.append(EntropyA[i]+EntropyB[i])

#The value of T where the maximum Entropy is seen is determined
    maxm = max(Entropy)
    for i in range(255):
        if (Entropy[i] == maxm):
            T=i
            
#Setting the value of the corresponding binary image less than and above the thresholds
    temp=deepcopy(I)
    for i in range(len(I[:,0])):
        for j in range(len(I[0,:])):
            if(I[i,j]>= T):
                temp[i,j]=255
            else:
                temp[i,j]=0
    print 'The thresholding value for the image is %s' %T

#Plot of the figure before and after thresholding    
    plt.figure()
    plt.subplot(2,1,1)
    plt.title('Before Thresholding')
    plt.imshow(I,cmap = cm.gray)
    plt.subplot(2,1,2)
    plt.title('After Thresholding')
    plt.imshow(temp,cmap = cm.gray)
    
    plt.show()

#Please provide the image path here
Image1=num.array(Image.open('C:/Users/Kiran/Desktop/Documents/UCF/Academics/Computer Vision/Assignment1/Berkley Dataset/157055.jpg'))
Image2=num.array(Image.open('C:/Users/Kiran/Desktop/Documents/UCF/Academics/Computer Vision/Assignment1/Berkley Dataset/43070.jpg'))
Image3=num.array(Image.open('C:/Users/Kiran/Desktop/Documents/UCF/Academics/Computer Vision/Assignment1/Berkley Dataset/147091.jpg'))

Thresholding(Image1)
Thresholding(Image2)
Thresholding(Image3)


