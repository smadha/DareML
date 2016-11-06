import numpy as np
import matplotlib.pyplot as plt
    
    
circle_data = np.genfromtxt("hw5_circle.csv", delimiter=",")
blob_data = np.genfromtxt("hw5_blob.csv", delimiter=",")



def plot_scatter(X_color, name, mean_points=[]):
    '''
    X_color = (list of points,"color")
    name = name/path of image
    mean_points = list of points
    '''
    if mean_points:
        mean_points = np.array(mean_points)
        plt.plot(mean_points[:,0], mean_points[:,1],"ro", markersize=10, color='black')
    
    for x_color in X_color:
        X = np.array(x_color[0])
        color = x_color[1]
        
        plt.plot(X[:,0], X[:,1],"ro", markersize=5, color=color)
        
    plt.figtext(.02, .02, name)
    plt.savefig(name)
    plt.close()
    
    
if __name__ == '__main__':
    plot_scatter([(circle_data,"red")], "circle_data_input")
    plot_scatter([(blob_data,"red")], "blob_data_input")
    
    