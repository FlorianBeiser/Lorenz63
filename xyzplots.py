# load python modules
import numpy as np
import matplotlib.pyplot as plt


class xyzplots:
    """ Common plot of true model, observations, and ensemble """

    def __init__(self, model, observation, ensemble):
        self.model = model
        self.observation = observation
        self.ensemble = ensemble
        
        
    def plot(self):
        """ Plotting """ 
        fig, axs = plt.subplots(3, sharex=True)
        axs[0].set_ylabel("x")
        axs[1].set_ylabel("y")
        axs[2].set_ylabel("z")
        axs[2].set_xlabel("t")
        self.model.plot(fig, axs)
        self.observation.plot(fig, axs)
        self.ensemble.plot(fig, axs)
        plt.show()
        