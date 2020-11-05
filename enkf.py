# load python modules
import numpy as np

class enkf:
    """ Ensemble Kalman Filter """ 
    
    def __init__(self, ensemble, observation, it_obs):
        self.ensemble = ensemble
        self.observation = observation
        self.it_obs = it_obs
        
        
    def assimilate(self):
        """ Assimilation of ensemble towards observation by EnKF """
        H = self.observation.H
        Pf = self.covState()
        print("Pf =", Pf)
        R = np.eye(3)
        
        S = np.dot(H, np.dot(Pf,H.T))
        print("S =", S)
        K = np.dot(Pf, np.dot(H.T, np.linalg.inv(S) ) )
        print("K = ", K)
                   
        y = self.observation.obs[:,it_obs]
                   
        for model in self.ensemble.ensemble:
            Xf = model.getCurrentState()
            d = y - np.dot(H, Xf)
             
            Xa = Xf + np.dot(K, d)
            ensemble.setCurrentState(Xa, Pa)

                   
    def covState(self):
        """ Calculates covariance of states """
        cov = np.zeros((3,3))
        mean = self.ensemble.getCurrentMean()
        print("mean =", mean)
        for model in self.ensemble.ensemble:
            state = model.getCurrentState()
            cov += np.outer( state - mean, state - mean )
        cov *= (1/(self.ensemble.Ne - 1))
        return cov