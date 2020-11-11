# load python modules
import numpy as np
import matplotlib.pyplot as plt


class observation:
    """ Observation of dynamical model """

    def __init__(self, dt, tmax, xobs=True, yobs=True, zobs=True):
        self.dt = dt 
        self.tmax = tmax
        
        # Determine dimension of observation
        self.dim = 0 
        if xobs == True:
            self.dim += 1
        if yobs == True:
            self.dim += 1
        if zobs == True:
            self.dim += 1
        assert(self.dim>0), "no observed variables"
        
        # Set up derived observation related arrays
        self.num = int(self.tmax/self.dt)
        self.obs = np.zeros((self.dim,self.num))
        
        self.times = np.arange(self.dt, self.tmax+self.dt, self.dt) # observation starts at t>0
        self.model_idx = np.zeros(self.num)

        # Construct observation matrix
        self.H = np.zeros((self.dim,3))
        x_already_observed = False
        y_already_observed = False
        z_already_observed = False
        for i in range(self.dim):
            if xobs == True and x_already_observed == False:
                self.H[i,0] = 1
                x_already_observed = True
                continue
            if yobs == True and y_already_observed == False:
                self.H[i,1] = 1
                y_already_observed = True
                continue
            if zobs == True and z_already_observed == False:
                self.H[i,2] = 1
                z_already_observed = True
                continue
        
        # Construct observation noise covariance matrix
        self.noise_level = 1.0
        self.R = self.noise_level * np.eye(self.dim)

        
    def noise(self):
        """ Returning realisation of observation error N(0,R) """
        return np.random.normal(0,self.noise_level,self.dim)
        
    def observe(self, model):
        """ observes model at observation times """
        self.model = model
        for i in range(self.num):
            self.model_idx[i] = int(self.times[i]/model.dt)
            self.obs[:,i] = np.dot(self.H, model.res[:,int(self.model_idx[i])])  + self.noise()
            
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

        i_matrix = 0
        for i in range(3):
            if np.sum(self.H[:,i])>0:
                x = self.obs[i_matrix,:]
                axs[i].plot( (np.arange(len(x))+1)*self.dt, x, "x", color="green" ) # observation does NOT start at t=0
                i_matrix += 1 
        
        if show == True:
            plt.setp(axs, xlim=(0,self.model.tmax))
            plt.show()
            