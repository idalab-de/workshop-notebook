import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.cbook import get_sample_data

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

def plot_decision_boundary(max_depth=1):
    '''
    This function plots the decision boundary for the iris dataset classified by a decision tree.
    '''
    # Parameters
    n_classes = 2
    images = ['orange.png', 'apple.png']
    plot_step = 0.02

    # Load data
    iris = load_iris()
    plt.figure(figsize=(30,15))

    for pairidx, pair in enumerate([[0, 1], 
#                                     [0, 2], [0, 3],
#                                     [1, 2], [1, 3], [2, 3]
                                   ]):
        # We only take the two corresponding features
        X = iris.data[:, pair]
        X[:,1] = X[:,1] +1.2
        X[:,0] = X[:,0] * 30
        y = iris.target
        X = X[y!=0,:]
        y = y[y!=0] -1 

        # Train
        clf = DecisionTreeClassifier(max_depth=max_depth).fit(X, y)

        # Plot the decision boundary
        plt.subplot(1, 1, pairidx + 1)

        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                             np.arange(y_min, y_max, plot_step))
        plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)

        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        cs = plt.contourf(xx, yy, Z, cmap=plt.cm.YlOrRd, vmin=-0.5, vmax=2.3)

        plt.xlabel('Weight [g]', fontsize = 20)
        plt.ylabel('Diameter [cm]', fontsize = 20)

        # Plot the training points
        for i, image in zip(range(n_classes), images):
            idx = np.where(y == i)
            imscatter(X[idx, 0][0], X[idx, 1][0], image)

#     plt.legend(loc='lower right', borderpad=0, handletextpad=0, prop={'size': 50})
    plt.axis("tight")
    plt.show()
    
    
def imscatter(x, y, image, ax=None, zoom=1):
    if ax is None:
        ax = plt.gca()
    try:
        image = plt.imread(image)
    except TypeError:
        # Likely already an array...
        pass
    im = OffsetImage(image, zoom=zoom)
    x, y = np.atleast_1d(x, y)
    artists = []
    for x0, y0 in zip(x, y):
        ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    return artists