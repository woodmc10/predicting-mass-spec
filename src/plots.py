import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import recall_score, f1_score
from data_class import Data, create_data
from model_class import Model


def pca_plots(df):
    '''Generate scree plot, 2D pca plot, and 3D pca plot for the data
    Paramters
    ---------
    df: DataFrame
        dataframe containing the reported target
    Return
    ------
    None
    '''
    # Prep data
    y = df.pop('reported')
    X = df.values
    scaler = StandardScaler()
    X_scale = scaler.fit_transform(X)

    # Scree Plot
    pca = PCA(n_components=4)
    X_pca = pca.fit_transform(X_scale)
    fig, ax = plt.subplots(figsize=(8, 8))
    scree_plot(ax, pca, 5, 'Scree Plot')

    # 2D PCA plot
    pca_2 = PCA(n_components=2)
    X_pca_2 = pca_2.fit_transform(X_scale)
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_mnist_embedding(ax, X_pca_2, y)
    plt.savefig('../images/pca_all_onehot_broad.png')
    plt.show()

    # 3D PCA plot
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
    ax.set_xlabel("Principal Component", fontsize=14)
    ax.set_ylabel("Variance Explained (%)", fontsize=14)
    ax.set_title('Principal Component Analysis', fontsize=16)


def plot_mnist_embedding(ax, X, y, tight=False, title=None):
    """Plot 2D pca.
    Parameters
    ----------
    ax: matplotlib.axis object
      The axis to make the scree plot on.
    X: numpy.array, shape (n, 2)
      A two dimensional array containing the coordinates of the embedding.
    y: numpy.array
      The labels of the datapoints.  Should be digits
    tight: bool
      If true use a tighter window to plot
    title: str
      A title for the plot.
    """
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    y_map = np.where(y == 1, '+', '-')
    y_c_map = np.where(y == 1, 'g', 'r')
    ax.patch.set_visible(False)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1],
                 y_map[i],
                 color=y_c_map[i],
                 fontdict={'weight': 'bold', 'size': 12},
                 alpha=0.3)
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_title('Principal Component Analysis', fontsize=16)
    if tight:
        ax.set_ylim([0, 0.4])
        ax.set_xlim([0, 0.4])
    else:
        ax.set_ylim([-0.1, 1.1])
        ax.set_xlim([-0.1, 1.1])


def pca_3d(fig, df, X_scale, y, pca_3):
    '''Plot 3D pca plot
    Parameters
    ----------
    fig: matplotlib.figure object
        figure to plot on
    df: DataFrame
        dataframe of data containing column names
    X_scale: numpy array
        scaled data from df
    y: numpy array
        targets
    pca_3: sklearn.decomposition.PCA object.
      A fit PCA object.
    '''
    result = pd.DataFrame(pca_3.transform(X_scale),
                          columns=['PCA%i' % i for i in range(3)],
                          index=df.index)
    y_c_map = np.where(y == 1, 'g', 'r')

    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(result['PCA0'], result['PCA1'], result['PCA2'],
               c=y_c_map, marker='x', alpha=0.3)

    xAxisLine = ((min(result['PCA0']), max(result['PCA0'])), (0, 0), (0, 0))
    ax.plot(xAxisLine[0], xAxisLine[1], xAxisLine[2], 'r')
    yAxisLine = ((0, 0), (min(result['PCA1']), max(result['PCA1'])), (0, 0))
    ax.plot(yAxisLine[0], yAxisLine[1], yAxisLine[2], 'r')
    zAxisLine = ((0, 0), (0, 0), (min(result['PCA2']), max(result['PCA2'])))
    ax.plot(zAxisLine[0], zAxisLine[1], zAxisLine[2], 'r')

    ax.set_xlabel('PCA0')
    ax.set_ylabel('PCA1')
    ax.set_zlabel('PCA2')
    ax.set_title('Principal Component Analysis', fontsize=16)
    plt.show()


def compound_bar_plot(df, save=False):
    ''''Plot a stacked bar plot with each bar representing the number of
    data points for each analyte separated into reported and unreported
    Parameters
    ----------
    df: DataFrame
        dataframe containing analyte column and reported column
        dataframe cannot be onehot encoded
    save: bool
        if true, save to images folder
    '''
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
    if save:
        plt.savefig('../images/compound_bar.png')


def scatter_plots(df, list_pairs, save=False):
    '''Generate four scatter plots from pairs of columns
    Parameters
    ----------
    df: DataFrame
        dataframe containing columns in list_pairs
    list_pairs:
        list of four pairs of columns to compare with scatter plots
    save: bool
        if true then save, if false then show plot
    '''
    fig = plt.figure(figsize=(10, 8))
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


def lasso_plot(data, pairs, save=False):
    '''Plot the lasso regularization of all the features at different
    learning rates.
    Parameters
    ----------
    data: Data class object
    save: bool
        If true, save figure. If false, show figure
    Return
    ------
    None
    '''
    X, y = Data.pop_reported(data.full_df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,
                                                        random_state=42)
    # Get weights from lasso model for 50 different alphas
    n_alphas = 50
    alphas = np.logspace(-6, -1, n_alphas)
    coefs = []
    for a in alphas:
        lasso = Lasso(alpha=a, fit_intercept=False, max_iter=100000)
        mod = Model(lasso, f1_score)
        mod.fit(X_train, y_train)
        coefs.append(mod.log_coef_())
    coefs = np.array(coefs).T
    labels = data.full_df.drop('reported', axis=1).columns
    
    # Get colors for plotting
    counter = 0
    cmap = plt.cm.get_cmap('tab20')
    color_list = []
    labels_list = []
    for index, label in enumerate(labels):
        if np.max(coefs[index]) > 0.2 or np.min(coefs[index]) < -0.2:
            color = counter * 2
            counter += 1
            lab = label
        else:
            color = 1
            lab = None
        color_list.append(color)
        labels_list.append(lab)

    # Zip the column names and weights together for plot labels
    zipped = zip(labels_list, coefs, color_list)
    zipped = list(zipped)
    sorted_zip = sorted(zipped, key=lambda x: np.abs(x[1][0]))

    # Plot all weights, keep less important weights in background/gray
    for i in range(len(pairs) + 1):
        if i < len(pairs):
            fig, ax = plt.subplots(figsize=(12, 8))
            for label, coef, color in sorted_zip:
                if label not in pairs[i]:
                    ax.plot(alphas, np.array(coef), label=None, c=cmap(1))
            for label, coef, color in sorted_zip:
                if label in pairs[i]:
                    ax.plot(alphas, np.array(coef), label=label, c=cmap(color))
        else:
            fig, ax = plt.subplots(figsize=(12, 8))
            for label, coef, color in sorted_zip:
                ax.plot(alphas, np.array(coef), label=label, c=cmap(color))
        ax.set_xscale('log')
        ax.set_xlabel('Learning Rate', fontsize=14)
        ax.set_ylabel('Coefficients', fontsize=14)
        ax.set_title('Lasso Feature Engineering', fontsize=16)
        ax.legend(loc=1)
        if save:
            plt.savefig('../images/lasso_pair' + str(i) + '.png')
        else:
            plt.show()


if __name__ == '__main__':

    all_df = create_data('../data/merged_df.csv', 'All')

    pairs = [('Relative Retention Time', 'Analyte Centroid Location (min)'),
             ('Analyte Peak Area (counts)', 'Analyte Peak Height (cps)'),
             ('area_ratio', 'height_ratio'),
             ('Analyte Peak Width (min)',
              'Analyte Peak Width at 50% Height (min)')]
    # scatter_plots(all_df.full_df, pairs, save=False)

    # pca_plots(all_df.full_df)

    lasso_plot(all_df, pairs, save=True)

    var_corr = all_df.limited_no_analyte_df.corr()
    sns.heatmap(var_corr)
