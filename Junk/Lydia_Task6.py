#Task 6. Download the image from the course page.
#done
#Resize the image to approximately 300 × 300 pixels and convert it to grayscale. 
from PIL import Image
import numpy as np
import scipy


image = Image.open("img.png")
new_size = 300
image = image.resize((new_size, new_size))
image = image.convert("L")  # Convert to grayscale

#Construct the blurring matrix A corresponding to the Gaussian kernel (1.9) in the 2D case.


#Lambda: 0<=lb<=1
#Number of samples: n
#def deblurr(lb = 0.05, nl = 0.5, n = 100, B = 1):
    #Constants

#Noise level for error
nl = 0.5
#Threshold, remove small entries
epsilon = 1e-9

n1 = 200 
n2 = 300
n = n1*n2

be1 = [-1, 1]                          #Bound x coord
be2 = [-1, 1]                          #Bound y coord
e1 = np.linspace(be1[0],be1[1],n1)     #x array (equidistant)
e2 = np.linspace(be2[0],be2[1],n2)     #y array (equidistant)
e1vv, e2vv = np.meshgrid(e1,e2,indexing='xy')         #2D coordinates of x,y. vx= all x val, vy= all y val
de1 = e1[1]-e1[0]                   #x step
de2 = e2[1]-e2[0]                   #y step

e1v = e1vv.flatten()  #Flattened x coordinates
e2v = e2vv.flatten()  #Flattened y coordinates

v = np.concatenate([e1v, e2v], axis=0).flatten()



#True signal = veck(X_true memb of R^nx,ny)
def xt(t):                       
    return np.exp(t)*np.sin(np.pi*t)

x = xt(v)       #x at coord v

#Gaussian kernel
B = 1   #In Gaussian kernel, choose sufficiently large. 1D
c = 200    #Constant
def K(s,t):     #Gaussian kernel (point spread function (PSF),defines the blur)
    return c*np.exp(-np.transpose(s-t) * B * (s-t)) #Assume shift invariant

#Blur matrix
A = np.zeros((n,n))
for i in range(n):
    for j in range(n):
        a = K(v[i],v[j]) * de1 * de2
        if np.abs(a) < epsilon: #Remove small entries
            A[i,j] = 0
        else:
            A[i,j] = a

#Error
np.random.seed(1477)
Ndist = np.random.normal(0, 1, n,)  #Normal distribution
e = nl * Ndist       #Noise

#Blurred noisy data
b = A @ x + e

