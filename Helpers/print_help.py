from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from Helpers.header import *

def categorical_scatter_2d(X2D, class_idxs, ms=3, ax=None, alpha=0.1, 
                           legend=True, figsize=None, show=False, 
                           savename=None, title=None, iters=None):
    ## Plot a 2D matrix with corresponding class labels: each class diff colour
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    #ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
    x = np.arange(10)
    classes = list(np.unique(class_idxs))
    ys = [i+x+(i*x)**2 for i in range(len(classes))]
    colors = plt.cm.rainbow(np.linspace(0,1,len(ys)))

    for i, cls in enumerate(classes):
        ax.plot(X2D[class_idxs==cls, 0], X2D[class_idxs==cls, 1], marker='o', 
            linestyle='', ms=ms, label=str(cls), alpha=alpha, color=colors[i],
            markeredgecolor='black', markeredgewidth=0.4)
    if legend:
        ax.legend()
        
    if savename is not None:
        plt.tight_layout()
        plt.savefig(savename)
    
    if title is not None:
        ax.set_title(title)
    
    if iters is not None:
        ax.set_xlabel(f'Iteration: {iters}')

    if show:
        plt.show()
    
    return ax

def plot_graph(train_PCA, labels, graph_label, dataset):
    """
    Plots the PCA results.
    """
    N = len(set(labels))
    fig, ax = plt.subplots(1, 1, figsize=(8,8))
    x = train_PCA[0]
    y = train_PCA[1]
    tag = labels

    cmap = plt.cm.jet
    cmpalist = [cmap(i) for i in range(cmap.N)]
    cmap = cmap.from_list('Custom cmap', cmpalist, cmap.N)

    bounds = np.linspace(0, N, N+1)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    scat = ax.scatter(x, y, c=tag, cmap=cmap, norm=norm)
    cb = plt.colorbar(scat, spacing='proportional', ticks=bounds)
    cb.set_label('Class Colours')
    ax.set_title(graph_label)
    plt.savefig(f'Plots/{dataset}_{graph_label}.jpeg')
    plt.show()