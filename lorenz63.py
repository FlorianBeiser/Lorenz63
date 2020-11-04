# load python modules
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class lorenz63:
    """ Lorenz 63 model """

    def __init__(self, p=10.0, r=28.0, b=8.0/3.0, tmax=100, dt=1e-3):
        # parameter for lorenz 63 system
        self.p = p
        self.r = r 
        self.b = b 
        # parameter for numerical method
        self.tmax = tmax
        self.dt = dt
        self.it = 0

                
    def simulate(self, X, T=None):
        """ Loranz 63 model simulated forward in time """
        assert(len(X)==3), "wrong size of initial X"
        self.X_init = X
        
        self.res = np.zeros( (3, int(self.tmax/self.dt)+1) )
        self.res[:,0] = X
        for i in range( int(self.tmax/self.dt) ):
            k0 = self.lorenz63f(X)
            k1 = self.lorenz63f( X + k0 * self.dt/2.0 )
            k2 = self.lorenz63f( X + k1 * self.dt/2.0)
            k3 = self.lorenz63f( X + k2 * self.dt )
            
            X += ( k0 + 2*k1 + 2*k2 + k3 ) * self.dt/6.0
            self.res[:,i+1] = X
    
    
    def lorenz63f(self, X):
        """ RHS of Lorenz system (R3 -> R3 independent of t) """
        return np.array([
                self.p * ( X[1] - X[0]), 
                X[0] * (self.r - X[2]) - X[1], 
                X[0] * X[1] - self.b * X[2]
                ])


    def plot3D(self):
        """ Plotting """
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_title("Lorenz '63 (RK4)")
        ax.plot(self.res[0,:], self.res[1,:], self.res[2,:], color="red", lw=1)
        plt.show()
    
    
    def plot(self):
        fig, axs = plt.subplots(3)
        for i in range(3):
            x = self.getTimeSeries(i)
            axs[i].plot( np.arange(len(x))*self.dt, x, color="red" )
        plt.show()
    
    
    def getLorenz63(self):
        return self.res
    
    
    def getTimeSeries(self,i):
        return self.res[i,:]