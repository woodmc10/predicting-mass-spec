import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.linear_model import Lasso
from sklearn.metrics import recall_score, f1_score, make_scorer, accuracy_score
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
        ax.text(X[i, 0], X[i, 1],
                y_map[i],
                color=y_c_map[i],
                fontdict={'weight': 'bold', 'size': 12},
                alpha=0.3)
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_title('Principal Component Analysis', fontsize=16)
    # red_minus = mpatches.Patch(color='red', hatch='-', alpha=0.3,
    #             label='Not Reported')
    # green_plus = mpatches.Patch(color='green', hatch='+', alpha=0.3,
    #              label='Reported')
    red_minus = Line2D([0], [0], color='red', ls='',
                       marker='_', alpha=0.3,
                       label='Not Reported')
    green_plus = Line2D([0], [0], color='g', ls='',
                        marker='+', alpha=0.3,
                        label='Reported')
    ax.legend(handles=[green_plus, red_minus])
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


def sort_columns(columns, coefs):
    '''Create lists containing column and value information that are
    sorted in the same order. In one list change the value of all
    analyte_* columns to 0. Update the column names to shorter names
    for including in plot labels
    Parameters
    ----------
    columns: list
        list of feature names, in order passed to model
    coefs: list
        list of coefs or feature importances in order returned from model
    Return
    ------
    sort_all_coefs: list
        coefs values sorted in order of abs(coef)
    sort_no_analyte_coefs: list
        coefs values sorted in same order as sort_all_coefs, but any coef
        from an analyte_ column is set to 0
    print_col: list
        list of column names formated for labels in same order as
        sort_all_coefs
    '''
    d_coefs = {columns[i]: coefs[i] for i in range(len(coefs))}
    sort_d_coefs = sorted(d_coefs, key=lambda k: abs(d_coefs[k]))
    sort_col = {'Analyte Peak Area (counts)': 'Peak Area',
                'Analyte Peak Height (cps)': 'Peak Height',
                'area_ratio': 'Area Ratio',
                'height_ratio': 'Height Ratio',
                'Analyte Start Time (min)': 'Start Time',
                'Anlayte Start Scan': 'Start Scan',
                'Analyte Stop Time (min)': 'Stop Time',
                'Analyte Centroid Location (min)': 'Centroid Location',
                'Analyte Stop Scan': 'Stop Scan',
                'Analyte Integration Quality': 'Integration Quality',
                'Analyte Peak Width at 50% Height (min)':
                'Peak Width 50% Height',
                'Analyte Peak Width (min)': 'Peak Width',
                'Analyte Peak Asymmetry': 'Peak Assymetry',
                'analyte_Azoxystrobin': 'Azoxystrobin',
                'analyte_Bifenazate': 'Bifenazate',
                'analyte_Etoxazole': 'Etoxazole',
                'analyte_Imazalil': 'Imazalil',
                'analyte_Imidacloprid': 'Imidacloprid',
                'analyte_Malathion': 'Malathion',
                'analyte_Myclobutanil': 'Myclobutanil',
                'analyte_Permethrin': 'Permethrin',
                'analyte_Spinosad': 'Spinosad',
                'analyte_Spiromesifen': 'Spiromesifen',
                'analyte_Spirotetramat': 'Spirotetramat',
                'analyte_Tebuconazole': 'Tebuconazole',
                'rt_diff': 'RT Difference',
                'baseline': 'Baseline Slope'}
    print_cols = []
    sort_all_coefs = []
    sort_no_analyte_coefs = []
    for key in sort_d_coefs:
        print_cols.append(sort_col.get(key, key))
        sort_all_coefs.append(d_coefs[key])
        if 'analyte_' in key:
            sort_no_analyte_coefs.append(0)
        else:
            sort_no_analyte_coefs.append(d_coefs[key])
    return sort_all_coefs, sort_no_analyte_coefs, print_cols


def feature_comparison(columns, coefs, features, label_values,
                       fig_name='plot.png', save=False):
    '''plot two bar charts, one for logistic regression coefficients,
    one for random forest feature importances
    Parameters
    ----------
    columns: list
        list of column names
    coefs: list
        list of coefficients from logistic regression
    features: list
        list of feature importances from random forest
    label_values: list of tuples
        list of titles for two plots, if logistic regression is included
        it must be the first of the two plots
    fig_name:
        file location for saving plot
    save: bool
        if true save plot, if false show plot
    Return
    ------
    None
    '''
    # Sort Column names and Feature Importances
    sort_coefs, sort_no_a_coefs, sort_d_coefs = sort_columns(columns, coefs)
    sort_feat, sort_no_a_feat, sort_d_feat = sort_columns(columns, features)

    # Formatting for Logistic Regression
    pos_coefs = [0 if coef < 0 else coef for coef in sort_coefs]
    pos_no_a_coefs = [0 if coef < 0 else coef for coef in sort_no_a_coefs]
    colors = ['r' if coef < 0 else 'g' for coef in sort_coefs]

    # Extract figure variables
    color1 = label_values[0][0]
    title1 = label_values[0][1]
    label1 = label_values[0][2]
    legend_title = None
    color2 = label_values[1][0]
    title2 = label_values[1][1]
    label2 = label_values[1][2]

    # Plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Left plot
    if title1 == 'Logistic Regression':
        colors = colors
    else:
        colors = color1
    importance_plot(axes[0], sort_coefs, sort_no_a_coefs, colors)
    importance_plot(axes[0], pos_coefs, pos_no_a_coefs, color1)
    axes[0].set_xlabel(label1, fontsize=14)
    axes[0].set_yticklabels(sort_d_coefs, fontsize=12)
    axes[0].set_title(title1, fontsize=16)
    axes[0].legend()
    handles, labels = axes[0].get_legend_handles_labels()
    order1 = [1, 0]

    # Add label to split out positive and negative coefficients
    if title1 == 'Logistic Regression':
        order2 = [3, 2]
        axes[0].legend([handles[idx] for idx in order2],
                       [labels[idx] for idx in order2],
                       loc='lower right', title='Positive Coefficients',
                       bbox_to_anchor=(1, 0.2))
        legend_title = 'Negative Coefficients'

    l1 = axes[0].legend([handles[idx] for idx in order1],
                        [labels[idx] for idx in order1],
                        loc='lower right', title=legend_title)
    axes[0].add_artist(l1)

    # Right plot
    importance_plot(axes[1], sort_feat, sort_no_a_feat, color2)
    axes[1].set_xlabel(label2, fontsize=14)
    axes[1].set_yticks(list(range(len(features))))
    axes[1].set_yticklabels(sort_d_feat, fontsize=12)
    axes[1].set_title(title2, fontsize=16)
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [1, 0]
    plt.legend([handles[idx] for idx in order],
               [labels[idx] for idx in order],
               loc='lower right')
    plt.tight_layout()
    if save:
        plt.savefig(fig_name)
    else:
        plt.show()


def importance_plot(ax, all_vals, no_analyte, colors):
    '''Add a bar plot of feature importance or coefficients to an axes
    Paramters
    ---------
    ax: matplotlib axes
        axes to add plot
    all_vals: list
        list containing all feature importance or coefficients
    no_analyte: list
        same list as all_vals but analyte columns have been set to 0
    colors: list or string
        list of colors, one for each value, or a single color
    Return
    ------
    ax: matplotlib axes
        axes with plot
    '''
    ax.barh(list(range(len(all_vals))), np.abs(all_vals), color=colors,
            alpha=0.1, label='Categorical Features')
    ax.barh(list(range(len(all_vals))), np.abs(no_analyte), color=colors,
            alpha=0.7, label='Continuous Features')
    ax.set_yticks(list(range(len(all_vals))))
    return ax


def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    axes : array of 3 axes, optional (default=None)
        Axes to use for plotting the curves.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    if axes is None:
        _, axes = plt.subplots(1, 1, figsize=(14, 5))

    axes.set_title(title)
    if ylim is not None:
        axes.set_ylim(*ylim)
    axes.set_xlabel("Training examples")
    axes.set_ylabel("F1 Score")

    scorer = make_scorer(f1_score)
    train_sizes, train_scores, test_scores = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes, scoring=scorer,
                       return_times=False, shuffle=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # Plot learning curve
    axes.grid()
    axes.fill_between(train_sizes, train_scores_mean - train_scores_std,
                      train_scores_mean + train_scores_std, alpha=0.1,
                      color="r")
    axes.fill_between(train_sizes, test_scores_mean - test_scores_std,
                      test_scores_mean + test_scores_std, alpha=0.1,
                      color="g")
    axes.plot(train_sizes, train_scores_mean, 'o-', color="r",
              label="Training F1 score")
    axes.plot(train_sizes, test_scores_mean, 'o-', color="g",
              label="Cross-validation F1 score")
    axes.legend(loc="best")


def confusion_plot(tp, fp, fn, tn, fig_name='plot.png', save=False):
    '''Plot the false positive and false negatives
    Parameters
    ----------
    fp: int
        number of false positive classifications
    fn: int
        number of false negative classifications
    fig_name: str
        file location to save plot
    save: bool
        if true save the plot, if false show the plot
    '''
    fig, ax = plt.subplots(1, 1)
    neg_rects = ax.bar([1, 2], [fp, tn], color=['r', 'r'], alpha=0.5, width=0.3,
                       align='edge', label='Not Reported')
    pos_rects = ax.bar([1, 2], [tp, fn], color=['g', 'g'], alpha=0.5, width=-0.3,
                       align='edge', label='Reported')
    ax.legend(title='Actual Outcome')
    ax.set_title('Classifications\n(Threshold = 0.55)', fontsize=16)
    ax.set_xlabel('Predicted Results', fontsize=14)
    ax.set_ylabel('Samples', fontsize=14)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Reported', 'Not Reported'])
    autolabel(neg_rects, ax)
    autolabel(pos_rects, ax)
    plt.tight_layout()
    if save:
        plt.savefig(fig_name)
    else:
        plt.show()


def autolabel(rects, ax):
    """
    Attach a text label above each bar displaying its height
    Parameters
    ----------
    rects: matplotlib rectangle object
        rectangle objects to label
    ax: matplotlib axis
        axis containing bar chart
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2., 1 * height,
                '%d' % int(height),
                ha='center', va='bottom')


def profit_curve(model, X_test, y_test, cost_matrix, ax, label, color=None,
                 fig_name='plot.png', save=False):
    ''' Plot a profit curve based
    Parameters
    ----------
    model: Model class
        model to generate predictions
    X_test: numpy array
        array of features
    y_test: numpy array
        array of targest
    cost_matrix:
        cost benefit matrix
        order: fp, tp, fn, tn
    ax: matplotlib ax
        ax to plot curve
    label: str
        label for curve
    color: str
        color for curve
    fig_name: str
        file name for saving plot
    save: bool
        if true save plot
    '''
    fpr, tpr, thresh = model.roc(X_test, y_test)
    thresh[0] = 1
    # convert fpr/tnr to include un-integrated samples
    tnr = 1 - fpr
    fpc = fpr * sum(y_test)
    tnc = tnr * sum(y_test)
    tnc = tnc + 27445
    nc = tnc + fpc
    tnr = tnc/nc
    fpr = fpc/nc

    fnr = 1 - tpr
    fpc, tpc, fnc, tnc = cost_matrix
    cost_arr = fpr * fpc + tpr * tpc + fnr * fnc + tnr * tnc
    cost_arr = cost_arr
    ax.plot(thresh, cost_arr, label=label, color=color)
    ax.set_title('Profit Curve')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Profit per Sample')
    ax.legend()
    if save:
        plt.savefig(fig_name)
    return ax


def stack_profit_curves(model_list, sample_list, mod, label_model_list,
                        label_sample_list, X_test, y_test, cost_matrix,
                        color_list, fig_name='plot.png', save=False):
    ''' Create two plots to compare profit curves, one comparing sampling
    approach, one comparing models
    Parameters
    ----------
    model_list: list
        list of models from Model class for comparison
    sample_list: list
        list of sampling types from Data class for comparison
    mod: Model class
        base model for sample type comparison
    label_model_list: list
        list of strings for labels in model comparison plot
    label_sample_list:
        list of strings for labels in sample comparison plot
    X_test: numpy array
        array of test set features for creating profit curve
    y_test: numpy array
        array of test set targets for creating profit curve
    cost_matrix: tuple
        floats describing the cost of each classification type
    color_list: list
        list of strings for colors in model comparison
    fig_name: str
        location to save plot
    save: bool
        if true save fig, if false show fig
    '''
    fig, axes = plt.subplots(2, 1, figsize=(6, 8))
    counter = 0
    for index, mod in enumerate(model_list):
        axes[0] = profit_curve(mod, X_test, y_test, cost_matrix, ax=axes[0],
                               label=label_model_list[index],
                               color=color_list[index])
    axes[0].set_title('Model Comparison (no minority sampling)')
    for X_train, y_train in sample_list:
        mod.fit(X_train, y_train)
        axes[1] = profit_curve(mod, X_test, y_test, cost_matrix, ax=axes[1],
                               label=label_sample_list[counter])
        counter += 1
    axes[1].set_title('Sampling Comparison (Tuned XGBoost)')
    X_train, y_train = sample_list[3]
    plt.tight_layout()
    if save:
        plt.savefig(fig_name)
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

    pca_plots(all_df.full_df)

    # lasso_plot(all_df, pairs)

    var_corr = all_df.full_df.corr()
    sns.heatmap(var_corr)
