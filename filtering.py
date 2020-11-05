# load python modules
import numpy as np
import matplotlib.pyplot as plt
import enkf

# for debugging
import importlib
importlib.reload( enkf ) 

class filtering:
    """ Ensemble of dynamical system models """ 
    
    def __init__(self, ensemble, observation, method="EnKF"):
        self.ensemble = ensemble
        self.ensemble.clearSimulations()
        self.observation = observation
        self.method = method

        
    def assimilate(self):
        """ Assimilation of ensemble towards observations """
        #for it_obs in range(self.observation.num):
        for it_obs in range(1):
            # Simulate until observation time
            t_obs = self.observation.times[it_obs]
            self.ensemble.simulate( t_obs )
            
            # Do EnKF
            #if self.method == "EnKF":
            #    assimilation = enkf.enkf(self.ensemble, self.observation, it_obs)
            #    assimilation.assimilate()
            
            
    def plot(self):
        """ Plotting """
        fig, axs = plt.subplots(3)
        self.ensemble.plot(fig, axs)
        self.observation.plot(fig, axs)