import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import plot_partial_dependence
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from data_class import Data, create_data
from model_class import Model


def variance_factor(df):
    '''Create a VIF dataframe for easy viewing of VIF
    Parameters
    ----------
    df: DataFrame
        dataframe containing column headers and target values (reported)
    Return
    ------
    sorted_vif: DataFrame
        dataframe with features sorted in order of VIF value
    '''
    X = df.drop('reported', axis=1)
    vif = pd.DataFrame()
    vif["VIF Factor"] = [variance_inflation_factor(X.values, i)
                         for i in range(X.shape[1])]
    vif["features"] = X.columns
    sorted_vif = vif.sort_values('VIF Factor', ascending=False)
    return sorted_vif


def compare_models(model_list, metric, X, y, thresh=0.5, plot=True):
    '''Generate ROC curves, threshold plots, and print out metric scores
    Parameters
    ----------
    model_list: list
        list of tuples (model, model name)
    metric: sklearn metric
        mestric used for scoring models
    X: np array
        array of features
    y: np array
        array of targets
    thresh: float
        defalut threshold for initial model evaluation
    plot: bool
        plot roc curve, threshold plot and determine best threhold for
        given metric if true
    Return
    ------
    metric_result: list
        list of tuples (model name, metric score)
    model_list
        original list of models
    '''
    metric_result = []
    fig, axes = plt.subplots(1, 2, figsize=(10, 6))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,
                                                        random_state=42)
    for model in model_list:
        mod = Model(model[0], metric)
        mod.fit(X_train, y_train)
        if plot:
            axes[0] = mod.roc_plot(X_test, y_test, axes[0], model[1])
            axes[1], thresh = mod.thresh_plot(X_test, y_test,
                                              axes[1], model[1])
        result = mod.score_metric(X_test, y_test, thresh)
        metric_result.append((model[1], result.round(2)))

        print(model[1])
        print(mod.summary(X_test, y_test))
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('ROC Curve')
    axes[0].legend()
    axes[1].set_xlabel('Threshold')
    axes[1].set_ylabel('F1 Score')
    axes[1].set_title('Threshold Comparison')
    plt.savefig('../images/roc_f1.png')
    plt.show()
    return metric_result, model_list


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
        print_cols.append(sort_col[key])
        sort_all_coefs.append(d_coefs[key])
        if 'analyte_' in key:
            sort_no_analyte_coefs.append(0)
        else:
            sort_no_analyte_coefs.append(d_coefs[key])
    return sort_all_coefs, sort_no_analyte_coefs, print_cols


def feature_comparison(columns, coefs, features, save=False):
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
    save: bool
        if true save plot, if false show plot
    Return
    ------
    None
    '''
    sort_coefs, sort_no_a_coefs, sort_d_coefs = sort_columns(columns, coefs)
    sort_feat, sort_no_a_feat, sort_d_feat = sort_columns(columns, features)
    colors = ['r' if coef < 0 else 'g' for coef in sort_coefs]
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    importance_plot(axes[0], sort_coefs, sort_no_a_coefs, colors)
    axes[0].set_xlabel('Coefficient Value', fontsize=14)
    axes[0].set_yticklabels(sort_d_coefs, fontsize=12)
    axes[0].set_title('Logistic Regression', fontsize=16)
    importance_plot(axes[1], sort_feat, sort_no_a_feat, 'orange')
    axes[1].set_xlabel('Feature Importance', fontsize=14)
    axes[1].set_yticks(list(range(len(features))))
    axes[1].set_yticklabels(sort_d_feat, fontsize=12)
    axes[1].set_title('Random Forest', fontsize=16)
    plt.tight_layout()
    if save:
        plt.savefig('../images/coef_features.png')
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
            alpha=0.2)
    ax.barh(list(range(len(all_vals))), np.abs(no_analyte), color=colors,
            alpha=0.7)
    ax.set_yticks(list(range(len(all_vals))))
    return ax


if __name__ == '__main__':
    all_df = create_data('../data/merged_df.csv', 'All')
    print(variance_factor(all_df.limited_df))

    # compare best models from random search
    X, y = Data.pop_reported(all_df.limited_df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,
                                                        stratify=y,
                                                        random_state=42)
    lr = LogisticRegression(penalty='none', class_weight='balanced',
                            solver='saga', max_iter=100000)
    rf = RandomForestClassifier(class_weight='balanced_subsample')
    mod_list = [(lr, 'Logistic Regression'), (rf, 'Random Forest')]
    scores, model_list = compare_models(mod_list, f1_score, X, y)
    print(scores)

    # plot feature importance and coefs
    log_complex = model_list[0][0]
    random_forest = model_list[1][0]
    coefs = log_complex.coef_[0]
    features = random_forest.feature_importances_
    columns = all_df.limited_df.drop('reported', axis=1).columns
    feature_comparison(columns, coefs, features, save=True)
