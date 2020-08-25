import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from data_class import Data

def pca_plots(df):
    y = df.pop('reported')
    X = df.values
    scaler = StandardScaler()
    X_scale = scaler.fit_transform(X)
    pca = PCA(n_components=4)
    X_pca = pca.fit_transform(X_scale)

    fig, ax = plt.subplots(figsize=(8, 8))
    scree_plot(ax, pca, 5, 'Scree Plot')

    pca_2 = PCA(n_components=2)
    X_pca_2 = pca_2.fit_transform(X_scale)
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_mnist_embedding(ax, X_pca_2, y, tight=False)
    # plt.savefig('../images/pca_all_onehot_broad.png')
    plt.show()

    pca_3 = PCA(n_components=3)
    pca_3.fit(X_scale)

    fig = plt.figure(figsize=(8, 8))
    pca_3d(fig, df, X_scale, y, pca_3)


def scree_plot(ax, pca, n_components_to_plot=8, title=None):
    """Make a scree plot showing the variance explained (i.e. variance
    of the projections) for the principal components in a fit sklearn
    PCA object.
    
    Parameters
    ----------
    ax: matplotlib.axis object
      The axis to make the scree plot on.
      
    pca: sklearn.decomposition.PCA object.
      A fit PCA object.
      
    n_components_to_plot: int
      The number of principal components to display in the scree plot.
      
    title: str
      A title for the scree plot.
    """
    num_components = pca.n_components_
    ind = np.arange(num_components)
    vals = pca.explained_variance_ratio_ * 100
    ax.plot(ind, vals, color='blue')
    ax.scatter(ind, vals, color='blue', s=50, alpha=0.5)

    for i in range(num_components):
        ax.annotate(r"{:2.2f}%".format(vals[i]), 
               (ind[i]+0.2, vals[i]+0.005), 
               va="bottom", 
               ha="center", 
               fontsize=12)

    ax.set_xticklabels(ind, fontsize=12)
    ax.set_ylim(0, max(vals) + 0.05)
    ax.set_xlim(0 - 0.45, n_components_to_plot + 0.45)
    ax.set_xlabel("Principal Component", fontsize=12)
    ax.set_ylabel("Variance Explained (%)", fontsize=12)
    if title is not None:
        ax.set_title(title, fontsize=16)


def plot_mnist_embedding(ax, X, y, tight=False, title=None):
    """Plot an embedding of the mnist dataset onto a plane.
    
    Parameters
    ----------
    ax: matplotlib.axis object
      The axis to make the scree plot on.
      
    X: numpy.array, shape (n, 2)
      A two dimensional array containing the coordinates of the embedding.
      
    y: numpy.array
      The labels of the datapoints.  Should be digits.

    tight: bool
      If true use a tighter window to plot
      
    title: str
      A title for the plot.
    """
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    y_map = np.where(y==1, '+', '-')
    y_c_map = np.where(y==1, 'g', 'r')
    ax.axis('off')
    ax.patch.set_visible(False)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], 
                 y_map[i], 
                 color=y_c_map[i], 
                 fontdict={'weight': 'bold', 'size': 12},
                 alpha=0.3)
    ax.set_xticks([]), 
    ax.set_yticks([])
    if tight:
        ax.set_ylim([0, 0.4]) 
        ax.set_xlim([0, 0.4]) 
    else:
        ax.set_ylim([-0.1, 1.1]) 
        ax.set_xlim([-0.1, 1.1])

    if title is not None:
        ax.set_title(title, fontsize=16)


def pca_3d(fig, df, X_scale, y, pca_3):
    result = pd.DataFrame(pca_3.transform(X_scale), columns=['PCA%i' % i for i in range(3)], index=df.index)
    y_map = np.where(y==1, '+', '_')
    y_c_map = np.where(y==1, 'g', 'r')

    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(result['PCA0'], result['PCA1'], result['PCA2'], c=y_c_map, marker='x', alpha=0.3)

    xAxisLine = ((min(result['PCA0']), max(result['PCA0'])), (0, 0), (0, 0))
    ax.plot(xAxisLine[0], xAxisLine[1], xAxisLine[2], 'r')
    yAxisLine = ((0, 0), (min(result['PCA1']), max(result['PCA1'])), (0, 0))
    ax.plot(yAxisLine[0], yAxisLine[1], yAxisLine[2], 'r')
    zAxisLine = ((0, 0), (0, 0), (min(result['PCA2']), max(result['PCA2'])))
    ax.plot(zAxisLine[0], zAxisLine[1], zAxisLine[2], 'r')

    ax.set_xlabel('PCA0')
    ax.set_ylabel('PCA1')
    ax.set_zlabel('PCA2')
    plt.show()


def compound_bar_plot(df, save=False):
    compound_group_df = df.groupby(['analyte', 'reported']).count()
    compound_group_df = compound_group_df['Sample Name']
    compound_group_df = compound_group_df.reset_index(level='reported')
    compound_group = pd.pivot(compound_group_df, columns='reported',
                              index=None)
    compound_group['total'] = (compound_group['Sample Name'][0]
                               + compound_group['Sample Name'][1])
    compound_group.sort_values('total', inplace=True, ascending=False)
    compound_group.pop('total')
    compound_group.plot(kind='bar', stacked=True, rot=45, color=['r', 'g'],
                        alpha=0.5, title='Chromatograms by Analyte')
    plt.xticks(horizontalalignment='right')
    plt.xlabel('Analyte')
    plt.ylabel('Chromatograms')
    plt.tight_layout()
    plt.legend(['Not Reported', 'Reported'])
    if save == True:
        plt.savefig('../images/compound_bar.png')


def scatter_plots(df, list_pairs, save=False):
    fig = plt.figure(figsize=(10, 8))
    # fig.suptitle('Scatter Plots')
    for num, pair in enumerate(list_pairs):
        ax = fig.add_subplot(2, 2, num + 1)
        x = df[pair[0]]
        y = df[pair[1]]
        ax.scatter(x, y, alpha=0.3)
        ax.set_xlabel(pair[0])
        ax.set_ylabel(pair[1])
        plt.tight_layout()
    if save:
        plt.savefig('../images/eda_four_scatter.png')
    else:
        plt.show()

if __name__ == '__main__':

    cols_drop_list = ['Analyte Start Scan', 'Analyte Stop Scan',
                        'Analyte Centroid Location (min)',
                        'Relative Retention Time',
                        'Analyte Integration Quality',
                        # 'Analyte Peak Area (counts)',
                        'Analyte Peak Height (cps)',
                        'Analyte Peak Width (min)',
                        'Analyte Peak Width at 50% Height (min)',
                        'height_ratio',
                        'area_ratio',
                        'Analyte Start Time (min)',
                        'Analyte Stop Time (min)']

    all_df = Data('../data/merged_df.csv', 'All', cols_drop_list)
    
    pairs = [('Relative Retention Time', 'Analyte Centroid Location (min)'),
             ('Analyte Peak Area (counts)', 'Analyte Peak Height (cps)'),
             ('area_ratio', 'height_ratio'), 
             ('Analyte Peak Width (min)',
              'Analyte Peak Width at 50% Height (min)')]
    scatter_plots(all_df.full_df, pairs, save=False)

    pca_plots(all_df.full_df)

    var_corr = all_df.limited_no_analyte_df.corr()
    sns.heatmap(var_corr)
    plt.show()
