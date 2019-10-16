import sys
from time import time

import matplotlib.pyplot as plt
import nlosbpy as bp
import numpy as np
from scipy.ndimage import convolve1d

from nlosrec.persistance import NLOSData

def main():
    cosine_pulse = np.array([0.01269278,0.00453593,-0.02004616,-0.05629666,-0.08190101,-0.06172356, 
                             0.02955968,0.17506646,0.29870288,0.29117546,0.08177177,-0.28399223, 
                             -0.62675316,-0.71654217,-0.4243675,0.15970903,0.74331225,0.99665849, 
                             0.7634847,0.16849521,-0.45986382,-0.79755001,-0.71654217,-0.33348839, 
                             0.09862947,0.36073404,0.38010257,0.22881968,0.03968434,-0.08511371, 
                             -0.11600235,-0.08190101,-0.02995483,0.00696196,0.02001015,0.01656925, 
                             0.00783852])

    chunk = [0,-1]
    k_size = 11

    if len(sys.argv) > 1:
        chunk = [10000,14000]
        ds = NLOSData(sys.argv[1])
        a = ds.data.reshape(np.prod(ds.data.shape[0:2]), ds.data.shape[-1]).copy()
        a = a[:, chunk[0]:chunk[1]]
        chunk = [0,-1]
        del ds
    else:
        grid_dim = [128,128]
        T = 128
        a = np.zeros((grid_dim[0]*grid_dim[1], T))
        a[:,:] = np.cos(np.linspace(0,np.pi,T)*8)
        """
        # Delta pulses
        a[:,:] = 0
        a[:,50] = 1
        a[:,0] = 1
        """

    print(a.shape)
    kernel = np.sin(np.linspace(-np.pi, +np.pi, k_size))
    kernel = kernel / kernel.max()
    #kernel[0:2] = -1
    #kernel[3:] = 1
    plt.plot(kernel)
    plt.figure()
    start = time()
    gpu_convolved = bp.convolve1d(a, kernel)
    gpu_time = time() - start

    start = time()
    sc_convolved = convolve1d(a, axis=-1, weights=kernel, mode='constant')
    sc_time = time() - start

    print("GPU Took %.5fs, SK Took %.5fs" % (gpu_time, sc_time))
    print(gpu_convolved[5, 0:10], gpu_convolved.shape)
    print(sc_convolved[5, 0:10], sc_convolved.shape)

    plt.plot(a[0, chunk[0]:chunk[1]], label='orig')
    plt.plot(sc_convolved[5, chunk[0]:chunk[1]], label='good')
    plt.plot(gpu_convolved[5, chunk[0]:chunk[1]], label='gpu')
    plt.plot(gpu_convolved[5, chunk[0]:chunk[1]]-sc_convolved[5,chunk[0]:chunk[1]], label='diff')
    plt.legend()

    plt.show()


if __name__ == '__main__':
    main()