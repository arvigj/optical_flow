import cv2
import numpy as np
import time
import itertools
from multiprocess import Pool
import matplotlib.pyplot as plt

camera = cv2.VideoCapture(0)
#camera.set(cv2.cv.CV_CAP_PROP_FPS, 100)

pool = Pool(4)

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

angles = [0,np.pi/6,np.pi/4,np.pi/3,np.pi/2]
sigmas = [2]

wavelets = map(lambda x: gabor(13,13,x[0],x[1]),itertools.product(sigmas,angles))
#plt.subplot(121)
#plt.imshow(wavelets[3].real)
#plt.subplot(122)
#plt.imshow(wavelets[3].imag)
#plt.show()
#print np.sum(wavelets[3])
#exit()

edges = np.zeros([len(wavelets),480,640],dtype=np.complex64)

time_now = time.time()
while(True):
    ret, cap = camera.read()
    cap = cv2.cvtColor(cap,cv2.COLOR_BGR2GRAY)
    edges.real = np.array(map(lambda data: cv2.filter2D(data[0],cv2.CV_64F,data[1].imag),zip(itertools.repeat(cap),wavelets)))
    edges.imag = np.array(map(lambda data: cv2.filter2D(data[0],cv2.CV_64F,data[1].imag),zip(itertools.repeat(cap),wavelets)))
    final = np.amax(np.abs(edges),axis=0)
    final = (255*(final/np.amax(final))).astype(np.uint8)
    cv2.imshow("image",final)
    if cv2.waitKey(1) == ord('q'):
        break
    print -1./(time_now-time.time())

    time_now = time.time()

camera.release()
cv2.destroyAllWindows()
