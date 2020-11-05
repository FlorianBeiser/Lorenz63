# load python modules
import copy 
import numpy as np
import matplotlib.pyplot as plt

class ensemble:
    """ Ensemble of dynamical system models """ 
    
    def __init__(self, model, Ne=10):
        self.model = model
        # ensemble parameters
        self.Ne = Ne
        self.ensemble = [None]*self.Ne
        # ensemble set with random perturbation in initial states
        for i in range(self.Ne):
            self.ensemble[i] = copy.deepcopy( model )
            self.ensemble[i].clearSimulation()
        self.setInitialStates()
        # ensemble properties
        self.meanT = None
        self.meant = np.zeros_like(np.array(self.ensemble[0].res))
            
    
    def setInitialStates(self, Cov=None):
        """ Setting initial state for all ensemble members """
        X0_model = self.model.X0
        np.random.seed(10000)
        for model in self.ensemble:
            if Cov is None:
                X0_member = X0_model + np.random.normal(0.0,1.0,3)
            else:
                X0_member = X0_model + np.random.multivariate_normal(np.zeros(3),Cov)
            model.setInitialState( X0_member )
    
            
    def setCurrentStates(self, mean, Cov=None):
        """" Setting current state for all ensemble members """
        for model in self.ensemble:
            if Cov is None:
                X = np.random.multivariate_normal( mean, Cov )
            else:
                X = np.random.multivariate_normal( mean, np.eye(3) )
            model.setCurrentState( X )
        

    def clearSimulations(self):
        """ Clears previous simulation results in the ensemble """
        self.meant = None
        self.meanT = None
        for model in self.ensemble:
            model.clearSimulation()

        
    def simulate(self, T=None):
        """ Simulates an ensemble of models forward in time with perturbed initial conditions """
        for model in self.ensemble:
            model.simulate( T ) 
        
        
    def getEnsemble(self):
        """ Returns ensemble """
        return self.ensemble
    
    
    def currentMean(self):
        """ Calculates mean of ensemble per timestep upto current time """
        it_model = self.ensemble[0].it
        print("it_model = ", it_model) 
        self.meant = np.zeros(3)
        print("res[:,+-1] = ", self.ensemble[0].res[:,it_model-1:it_model+1])
        for model in self.ensemble:
            self.meant += model.res[:,it_model]
        self.meant /= self.Ne
        

    def getCurrentMean(self):
        """ Returns mean of ensemble per timestep upto current time """
        self.currentMean()
        return self.meant
            

    def mean(self):
        """ Calculates mean of ensemble per timestep for full time series """
        self.meanT = np.zeros_like(np.array(self.ensemble[0].res))
        for model in self.ensemble:
            self.meanT += model.res
        self.meanT /= self.Ne
        

    def plot(self, fig=None, axs=None):
        """ Plotting """
        # Plotting mean
        if fig is None:
            fig, axs = plt.subplots(3)
            show = True
        else:
            show = False

        if self.meanT is not None:
            for i in range(3):
                x = self.meanT[i,:]
                axs[i].plot(np.arange(len(x))*self.model.dt,x, color="blue")
        
        for model in self.ensemble:
            for i in range(3):
                x = model.getTimeSeries(i)
                axs[i].plot(np.arange(len(x))*self.model.dt,x, color="blue",alpha=0.1)
        
        if show == True:
            plt.show()
            
    