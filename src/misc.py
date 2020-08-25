import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score, f1_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from statsmodels.stats.outliers_influence import variance_inflation_factor
from plots import compound_bar_plot, scree_plot, plot_mnist_embedding, scatter_plots
from data_class import Data
from logistic_class import Model


def variance_factor(df):
    X = df.drop('reported', axis=1)
    vif = pd.DataFrame()
    vif["VIF Factor"] = [variance_inflation_factor(X.values, i)
                         for i in range(X.shape[1])]
    vif["features"] = X.columns
    sorted_vif = vif.sort_values('VIF Factor', ascending=False)
    return sorted_vif

def compare_models(model_list, metric, X, y, plot=True):
    metric_result = []
    fig, ax = plt.subplots()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,
                                                        random_state=42)
    for model in model_list:
        mod = Model(model[0], metric)
        mod.fit(X_train, y_train)
        result = mod.score_metric(X_test, y_test)
        metric_result.append((model[1], result.round(2)))
        if plot:
            ax = mod.roc_plot(X_test, y_test, ax, model[1])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend()
    plt.show()
    return metric_result

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

    X, y = Data.pop_reported(all_df.full_df)
    lr_simp = LogisticRegression()
    lr = LogisticRegression(penalty='l1', class_weight='balanced',
                            solver='liblinear')
    rf = RandomForestClassifier(class_weight='balanced_subsample')
    mod_list = [(lr_simp, 'Log Standard'), (lr, 'Log L1 Balanced'),
                (rf, 'Rand Forest')]
    print(compare_models(mod_list, f1_score, X, y))

    '''
    zipped = zip(log_complex.log_coef_()[0],
                 all_df.limited_df.drop('reported', axis=1).columns)
    for coef, column in zipped:
        print(f'{column}: {coef:.2f}')


    fig, ax = plt.subplots()
    ax = log_complex.thresh_plot(X_test, y_test, ax, 'Log_L1')
    ax = log_simp.thresh_plot(X_test, y_test, ax, 'Log')
    ax = rand_forest.thresh_plot(X_test, y_test, ax, 'Random Forest')
    ax.set_xlabel('Theshold')
    ax.set_ylabel('F1 Score')
    ax.legend()
    plt.show()
    '''
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,
                                                        random_state=42)

    n_alphas = 50
    alphas = np.logspace(-6, -1, n_alphas)

    coefs = []
    for a in alphas:
        lasso = Lasso(alpha=a, fit_intercept=False, max_iter=10000)
        mod = Model(lasso, f1_score)
        mod.fit(X_train, y_train)
        coefs.append(mod.log_coef_())
    breakpoint()
    fig, ax = plt.subplots()
    for i in range(len(coefs[0])-1):
        label = all_df.full_df.drop('reported', axis=1).columns[i]
        ax.plot(alphas, np.array(coefs)[:, i], label=label)
    ax.set_xscale('log')
    # ax.set_xlim(ax.get_xlim()[::-1])
    ax.set_xlabel('alpha')
    ax.set_ylabel('coefs')
    ax.legend()
    plt.show()
    