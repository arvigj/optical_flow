import cv2
import numpy as np
import itertools
import time
import matplotlib.pyplot as plt
from numpy import linalg as LA



def gabor(x_range, y_range, sigma, theta):
    sigma = float(sigma)
    theta = float(theta)
    dot = lambda x: np.dot([x[0],x[1]],[np.cos(theta),np.sin(theta)])
    gaussian = lambda x: np.exp(-(x[0]**2 + x[1]**2)/(2*sigma**2))
    phase = lambda x: np.exp(1j*np.pi*dot(x)/(2*sigma))

    c2_num =sum(map(lambda x: gaussian(x)*phase(x),itertools.product(xrange(-x_range/2,x_range/2+1),xrange(-y_range/2,y_range/2+1))))
    c2_denom =sum(map(lambda x: gaussian(x),itertools.product(xrange(-x_range/2,x_range/2+1),xrange(-y_range/2,y_range/2+1))))
    c2 = c2_num/c2_denom

    Z_fun = lambda x: (1-2*c2*np.cos(np.pi*dot(x)/(2*sigma))+c2**2)*np.exp(-(x[0]**2+x[1]**2)/(sigma**2))
    c1 = sigma/np.sqrt(sum(map(Z_fun,itertools.product(xrange(-x_range/2,x_range/2+1),xrange(-y_range/2,y_range/2+1)))))

    psi = lambda x: (c1/sigma)*(phase(x)-c2)*gaussian(x)
    psi_array = np.zeros([x_range,y_range],dtype="complex")
    def fill(pos):
        x_centered = pos[0]-(x_range/2)
        y_centered = pos[1]-(y_range/2)
        psi_array[pos[0],pos[1]] = psi([x_centered,y_centered])
    map(fill,itertools.product(xrange(x_range),xrange(y_range)))
    return psi_array

class Flow:
    def __init__(self,sigma):
        self.angles = np.array([0, np.pi/4, np.pi/2, 3*np.pi/4])
        self.wavelets = map(lambda x: gabor(13,13,sigma,self.angles[x]).imag,xrange(len(self.angles)))
        self.sigma = sigma
        self.dx = None
        self.dx = None
        self.dt = None
        self.W = None
        self.image1 = None
        self.image2 = None
    def flow(self, image1, image2):
        self.image1 = cv2.GaussianBlur(image1,(self.sigma,self.sigma),0)[::self.sigma,::self.sigma].astype(np.float64)
        self.image2 = cv2.GaussianBlur(image2,(self.sigma,self.sigma),0)[::self.sigma,::self.sigma].astype(np.float64)
        convolutions = np.array(map(lambda x: cv2.filter2D(self.image1, cv2.CV_64F, self.wavelets[x]),xrange(len(self.wavelets))))
        theta_max = np.argmax(np.abs(convolutions),axis=0)
        angle_matrix = self.angles[theta_max]
        self.W = np.zeros(convolutions.shape[1:])
        for i in xrange(self.W.shape[0]):
            for j in xrange(self.W.shape[1]):
                self.W[i,j] = convolutions[theta_max[i,j],i,j]
        self.dx = np.multiply(self.W,np.cos(angle_matrix))
        self.dy = np.multiply(self.W,np.sin(angle_matrix))
        self.dt = self.image2 - self.image1
        f = map(lambda xy: self.eig(xy[0],xy[1]),itertools.product(xrange(1,self.image1.shape[0]-1),xrange(1,self.image1.shape[1]-1)))
        sy, sx = self.image1.shape
        f = np.array(f).reshape([sy-2,sx-2,2,1])
        print f[:,:,0].shape
        return f[:,:,1,0]

    def eig(self, y, x, eta=0.01, delta=0.01):
        if (np.amax(self.W[y-1:y+2,x-1:x+2]) < eta) or (np.amax(self.dt[y-1:y+2,x-1:x+2]) < delta):
            return np.zeros([2,1])
        A = np.concatenate((self.dx[y-1:y+2,x-1:x+2].ravel()[:,np.newaxis].T,self.dy[y-1:y+2,x-1:x+2].ravel()[:,np.newaxis].T),axis=0)
        ATA = np.dot(A,A.T)
        eigvals, eigvec = LA.eig(ATA)
        if np.amin(eigvals) < 0.03:
            return np.zeros([2,1])
        V = np.dot(LA.inv(ATA),A)
        V = np.dot(V, self.dt[y-1:y+2,x-1:x+2].flatten()[:,np.newaxis])
        return V





camera = cv2.VideoCapture(0)
flow = Flow(31)

image1 = np.zeros([480,640]) + 255
image1[240-50:240+50,320-50:320+50] = 0
image2 = np.zeros([480,640]) + 255
image2[240-80:240+20,320-80:320+20] = 0


#time.sleep(1)
#retval, image1 = camera.read()
#image1 = cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)
#retval, image2 = camera.read()
#image2 = cv2.cvtColor(image2,cv2.COLOR_BGR2GRAY)

plt.imshow(image1,cmap="gray")
plt.show()
plt.imshow(image2,cmap="gray")
plt.show()
plt.imshow(flow.flow(image1,image2),cmap="gray")
plt.show()

