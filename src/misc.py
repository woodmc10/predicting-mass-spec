import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score, f1_score, roc_curve, plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import plot_partial_dependence
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from plots import compound_bar_plot, scree_plot, plot_mnist_embedding, scatter_plots
from data_class import Data
from model_class import Model


def variance_factor(df):
    X = df.drop('reported', axis=1)
    vif = pd.DataFrame()
    vif["VIF Factor"] = [variance_inflation_factor(X.values, i)
                         for i in range(X.shape[1])]
    vif["features"] = X.columns
    sorted_vif = vif.sort_values('VIF Factor', ascending=False)
    return sorted_vif

def compare_models(model_list, metric, X, y, thresh=0.5, plot=True):
    metric_result = []
    fig, axes = plt.subplots(1, 2, figsize=(10, 6))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,
                                                        random_state=42)
    for model in model_list:
        mod = Model(model[0], metric)
        mod.fit(X_train, y_train)
        if plot:
            axes[0] = mod.roc_plot(X_test, y_test, axes[0], model[1])
            axes[1], thresh = mod.thresh_plot(X_test, y_test, axes[1], model[1])
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
    # plt.savefig('../images/roc_f1.png')
    return metric_result, model_list


def sort_columns(columns, coefs):
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

def importance_plot(ax, all_vals, no_analyte, colors):
    ax.barh(list(range(len(all_vals))), np.abs(all_vals), color=colors,
                 alpha=0.2)
    ax.barh(list(range(len(all_vals))), np.abs(no_analyte), color=colors,
                 alpha=0.7)
    ax.set_yticks(list(range(len(all_vals))))
    return ax

if __name__ == '__main__':

    cols_drop_list = ['Analyte Start Scan', 'Analyte Stop Scan',
                        'Analyte Centroid Location (min)',
                        'Relative Retention Time',
                        'Analyte Integration Quality',
                        'Analyte Peak Height (cps)',
                        'Analyte Peak Width at 50% Height (min)',
                        'height_ratio',
                        'area_ratio',
                        'Analyte Start Time (min)',
                        'Analyte Stop Time (min)']

    all_df = Data('../data/merged_df.csv', 'All', cols_drop_list)
    # print(variance_factor(all_df.limited_df))

    X, y = Data.pop_reported(all_df.limited_df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,
                                                        stratify=y,
                                                        random_state=42)
    lr = LogisticRegression(penalty='none', class_weight='balanced',
                            solver='saga', max_iter=100000)
    rf = RandomForestClassifier(class_weight='balanced_subsample')
                                # min_samples_leaf=0.0043,
                                # min_samples_split=7,
                                # n_estimators=200)
    mod_list = [(lr, 'Log L1 Balanced'), (rf, 'Rand Forest')]
    scores, model_list = compare_models(mod_list, f1_score, X, y)
    print(scores)

    log_complex = model_list[0][0]
    random_forest = model_list[1][0]
    
    
    coefs = log_complex.coef_[0]
    features = random_forest.feature_importances_
    columns = all_df.limited_df.drop('reported', axis=1).columns
    zipped = zip(coefs, features, columns)
    print('Feature: Coef, Importance')
    for coef, feature, column in zipped:
        print(f'{column}: {coef:.2f}, {feature:.2f}')
    
    
    sort_coefs, sort_no_a_coefs, sort_d_coefs = sort_columns(columns, coefs)
    sort_feat, sort_no_a_feat, sort_d_feat = sort_columns(columns, features)
    colors = ['r' if coef < 0 else 'g' for coef in sort_coefs]
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    importance_plot(axes[0], sort_coefs, sort_no_a_coefs, colors)
    axes[0].set_xlabel('Coefficient Value')
    axes[0].set_yticklabels(sort_d_coefs)
    axes[0].set_title('Logistic Regression')
    importance_plot(axes[1], sort_feat, sort_no_a_feat, 'orange')
    axes[1].set_xlabel('Feature Importance')
    axes[1].set_yticks(list(range(len(features))))
    axes[1].set_yticklabels(sort_d_feat)
    axes[1].set_title('Random Forest')
    plt.tight_layout()
    plt.savefig('../images/coef_features.png')
    plt.show()
    