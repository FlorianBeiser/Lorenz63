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
        
        self.times = np.arange(self.dt, self.tmax+self.dt, self.dt) # observation starts at t>0
        self.model_idx = np.zeros(self.num)
        
        self.noise_level = 1.0
        self.R = self.noise_level * np.eye(3)
        self.H = np.eye(3)
        
        
    def noise(self):
        """ Returning realisation of observation error N(0,R) """
        return np.random.normal(0,self.noise_level,3)
        
    def observe(self, model):
        """ observes model at observation times """
        self.model = model
        np.random.seed(0)
        for i in range(self.num):
            self.model_idx[i] = int(self.times[i]/model.dt)
            self.obs[:,i] = model.res[:,int(self.model_idx[i])]  + self.noise()
            
    def getObservation(self):
        """ Returns observations """
        return self.obs
            
    def plot(self, fig = None, axs = None):
        """ Plotting """
        if fig is None:
            fig, axs = plt.subplots(3, sharex=True)
            show = True
        else:
            show = False

        for i in range(3):
            x = self.obs[i,:]
            axs[i].plot( (np.arange(len(x))+1)*self.dt, x, "x", color="green" ) # observation does NOT start at t=0
        
        if show == True:
            plt.setp(axs, xlim=(0,self.model.tmax))
            plt.show()
            