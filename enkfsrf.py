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
        HXf = np.zeros((self.observation.Ny, self.ensemble.Ne))
        for i in range(self.ensemble.Ne):
            HXf[:,i] = np.dot(H,Xf[:,i]) 
            
        HXfmean = np.sum(HXf,axis=1)/self.ensemble.Ne
        HXfmean.shape = (self.observation.Ny,1)
        
        S = HXf - HXfmean 
        
        SS = 1/(self.ensemble.Ne-1) * np.dot(S,S.T)

        F = SS + self.observation.R
        Finv = np.linalg.inv(F)
        
        y = self.observation.obs[:,self.it_obs]
        y.shape = (self.observation.Ny,1)
        
        Y = np.dot(y,np.ones((1,self.ensemble.Ne)))
        Ypert = np.zeros((self.observation.Ny,self.ensemble.Ne))
        for i in range(self.ensemble.Ne):
            Ypert[:,i] = self.observation.noise()
        
        D = Y - HXf + Ypert
        
        C = np.dot(Finv,D)
        
        E = np.dot(S.T, C)
        
        Xa = Xf + 1/(self.ensemble.Ne-1) * np.dot(Xfpert,E) 
        
        for i in range(self.ensemble.Ne):
            self.ensemble.ensemble[i].setCurrentState(Xa[:,i])
            
            
            