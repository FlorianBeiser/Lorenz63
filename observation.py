# load python modules
import numpy as np
import matplotlib.pyplot as plt


class observation:
    """ Observation of dynamical model """

    def __init__(self, dt, tmax):
        self.dt = dt 
        self.tmax = tmax
        # Set up derived observation related arrays
        self.num = int(self.tmax/self.dt)
        self.obs = np.zeros((3,self.num))
        
        self.times = np.arange(self.dt, self.tmax+self.dt, self.dt)
        self.model_idx = np.zeros(self.num)
        
    def observe(self, model):
        """ observes model at observation times """
        for i in range(self.num):
            self.model_idx[i] = int(self.times[i]/model.dt)
            self.obs[:,i] = model.res[:,int(self.model_idx[i])]  + np.random.normal(0,1,3)
            
    def getObservation(self):
        """ Returns observations """
        return self.obs
            
    def plot(self):
        """ Plotting """
        fig, axs = plt.subplots(3)
        for i in range(3):
            x = self.obs[i,:]
            axs[i].plot( np.arange(len(x))*self.dt, x, "x", color="red" )
        plt.show()
            