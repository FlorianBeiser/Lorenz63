# load python modules
import copy 
import numpy as np
import matplotlib.pyplot as plt

class ensemble:
    """ Ensemble of dynamical system models """ 
    
    def __init__(self, model, Ne=10):
        self.Ne = Ne
        self.ensemble = [None]*self.Ne
        self.model = model
        for i in range(self.Ne):
            self.ensemble[i] = copy.deepcopy(model)
        self.meant = None
        
    def simulate(self):
        """ Simulates an ensemble of models forward in time with perturbed initial conditions """
        np.random.seed(10000)
        for model in self.ensemble:
            X = np.random.normal(1.0,1.0,3)
            model.simulate( X )
        
        
    def getEnsemble(self):
        """ Returns ensemble """
        return self.ensemble
            

    def mean(self):
        """ Calculats mean of ensemble per timestep """
        self.meant = np.zeros_like(np.array(self.ensemble[0].res))
        for model in self.ensemble:
            self.meant += model.res
        self.meant /= self.Ne
        

    def plot(self, fig=None, axs=None):
        """ Plotting """
        # Plotting mean
        if fig is None:
            fig, axs = plt.subplots(3)
            show = True
        else:
            show = False

        if self.meant is not None:
            for i in range(3):
                x = self.meant[i,:]
                axs[i].plot(np.arange(len(x))*self.model.dt,x, color="blue")
        
        for model in self.ensemble:
            for i in range(3):
                x = model.getTimeSeries(i)
                axs[i].plot(np.arange(len(x))*self.model.dt,x, color="blue",alpha=0.1)
        
        if show == True:
            plt.show()
            
    