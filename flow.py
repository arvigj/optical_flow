import cv2
import numpy as np
import itertools
import time
import matplotlib.pyplot as plt



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
    def flow(self, image1, image2):
        image1 = cv2.GaussianBlur(image1,(self.sigma,self.sigma),0)[::self.sigma,::self.sigma]
        image2 = cv2.GaussianBlur(image2,(self.sigma,self.sigma),0)[::self.sigma,::self.sigma]
        convolutions = np.array(map(lambda x: cv2.filter2D(image1, cv2.CV_64F, self.wavelets[x]),xrange(len(self.wavelets))))
        theta_max = np.argmax(np.abs(convolutions),axis=0)
        angle_matrix = self.angles[theta_max]
        W = np.zeros(convolutions.shape[1:])
        for i in xrange(W.shape[0]):
            for j in xrange(W.shape[1]):
                W[i,j] = convolutions[theta_max[i,j],i,j]
        dx = np.multiply(W,np.cos(angle_matrix))
        dy = np.multiply(W,np.sin(angle_matrix))
        dt = image2 - image1
        print W
        return dt

camera = cv2.VideoCapture(0)
flow = Flow(5)

time.sleep(1)
retval, image1 = camera.read()
image1 = cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)
retval, image2 = camera.read()
image2 = cv2.cvtColor(image2,cv2.COLOR_BGR2GRAY)

plt.imshow(flow.flow(image1,image2),cmap="gray")
plt.show()

