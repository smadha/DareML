import numpy as np

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt

def save_3d_fig(acc_data, fig_name,labels=["X","Y","Z"] ):
    acc_data=np.array(acc_data)
    max_val = max(acc_data[:,2])
    print ""
    print fig_name,"Max-", max_val,labels, " = ", acc_data[np.where(acc_data[:,2] == max_val)[0][0]]
    print ""
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X, Y, Z = acc_data[:,0], acc_data[:,1], acc_data[:,2]
    X, Y = np.meshgrid(X, Y)
    
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    # ax.set_zlim(-1.01, 1.01)
    
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_zlabel(labels[2])
    
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.figtext(.02, .02, fig_name)
    
    plt.savefig(fig_name)
    plt.close()
    
def save_2d_fig(acc_data, fig_name,labels=["X","Y"] ):
    acc_data=np.array(acc_data)
    max_val = max(acc_data[:,1])
    print ""
    print fig_name,"Max-", max_val,labels,"=", acc_data[np.where(acc_data[:,1] == max_val)[0][0]]
    print ""
    plt.plot(acc_data[:,0], acc_data[:,1])
    plt.figtext(.02, .02, fig_name)
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.savefig(fig_name)
    plt.close()

if __name__ == '__main__':
    save_3d_fig([[-6, 1, 55.75], [-5, 2, 55.75], [-4, 3, 55.75], [-3, 4, 73.0], [-2, 5, 92.2], [-1, 6, 93.1], [0, 7, 95.45], [1, 8, 95.65], [2, 9, 96.85]], "fig_name")

