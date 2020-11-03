# load python modules
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


    def simulate(self, X, dt=None):
        """ Loranz 63 model simulated forward in time """
        assert(len(X)==3), "wrong size of initial X"
        
        self.res = [[X[0]],[X[1]],[X[2]]]
        for _ in range( int(self.tmax/self.dt) ):
            k_0 = self.lorenz63f(X)
            k_1 = self.lorenz63f([ x + k * self.dt / 2 for x, k in zip(X, k_0) ])
            k_2 = self.lorenz63f([ x + k * self.dt / 2 for x, k in zip(X, k_1) ])
            k_3 = self.lorenz63f([ x + k * self.dt for x, k in zip(X, k_2) ])
            for i in range(3):
                X[i] += (k_0[i] + 2 * k_1[i] + 2 * k_2[i] + k_3[i]) * self.dt / 6.0
                self.res[i].append(X[i])
    
    
    def lorenz63f(self, X):
        """ Lorenz equation """
        return [
                -self.p * X[0] + self.p * X[1], 
                -X[0] * X[2] + self.r * X[0] - X[1], 
                X[0] * X[1] - self.b * X[2]
                ]


    def plot3D(self):
        """ Plotting """
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_title("Lorenz '63 (RK4)")
        ax.plot(self.res[0], self.res[1], self.res[2], color="red", lw=1)
        plt.show()
        
    def plot(self):
        fig, axs = plt.subplots(3)
        for i in range(3):
            x = self.getTimeSeries(i)
            axs[i].plot( range(len(x)), x )
    
    
    def getTimeSeries(self):
        return self.res
    
    
    def getTimeSeries(self,i):
        return self.res[i]