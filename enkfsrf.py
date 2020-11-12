# load python modules
import numpy as np

class enkfsrf:
    """ Ensemble Kalman Filter in Square Root Formulation """ 
    
    def __init__(self, ensemble, observation, it_obs):
        self.ensemble = ensemble
        self.observation = observation
        self.it_obs = it_obs
        
        
    def assimilate(self):
        """ Assimilation of ensemble towards observation by EnKF in square root formulation """
        """ Notation following Vetra-Carvalho et al 2018 """
        
        Xfmean = self.ensemble.getCurrentMean()
        Xfmean.shape = (3,1)
        
        Xf = np.zeros((3,self.ensemble.Ne))
        for i in range(self.ensemble.Ne):
            Xf[:,i] = self.ensemble.ensemble[i].getCurrentState()
            
        Xfpert = Xf - Xfmean
                
        H = self.observation.H        
        HX = np.zeros((self.observation.Ny, self.ensemble.Ne))
        for i in range(self.ensemble.Ne):
            HX[:,i] = np.dot(H,Xf[:,i]) + self.observation.noise()
            
        HXmean = np.sum(HX,axis=1)/self.ensemble.Ne
        HXmean.shape = (self.observation.Ny,1)
        
        S = HX - HXmean
        
        SS = 1/(self.ensemble.Ne-1) * np.dot(S,S.T)

        F = SS + self.observation.R
        Finv = np.linalg.inv(F)
        
        y = self.observation.obs[:,self.it_obs]
        y.shape = (self.observation.Ny,1)
        
        Y = np.dot(y,np.ones((1,self.ensemble.Ne)))
        
        D = Y - HX 
        
        C = np.dot(Finv,D)
        
        E = np.dot(S.T, C)
        
        Xa = Xf + 1/(self.ensemble.Ne-1) * np.dot(Xfpert,E) 
        
        for i in range(self.ensemble.Ne):
            self.ensemble.ensemble[i].setCurrentState(Xa[:,i])
            
            
            
            
            