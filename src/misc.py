import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score, f1_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import plot_partial_dependence
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
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('ROC Curve')
    axes[0].legend()
    axes[1].set_xlabel('Threshold')
    axes[1].set_ylabel('F1 Score')
    axes[1].set_title('Threshold Comparison')
    # plt.savefig('../images/roc_f1.png')
    return metric_result, model_list

if __name__ == '__main__':

    cols_drop_list = ['Analyte Start Scan', 'Analyte Stop Scan',
                        'Analyte Centroid Location (min)',
                        'Relative Retention Time',
                        'Analyte Integration Quality',
                        # # 'Analyte Peak Area (counts)',
                        'Analyte Peak Height (cps)',
                        # 'Analyte Peak Width (min)',
                        'Analyte Peak Width at 50% Height (min)',
                        'height_ratio',
                        'area_ratio',
                        'Analyte Start Time (min)',
                        'Analyte Stop Time (min)']

    all_df = Data('../data/merged_df.csv', 'All', cols_drop_list)
    # print(variance_factor(all_df.limited_df))

    X, y = Data.pop_reported(all_df.limited_df)
    lr = LogisticRegression(penalty='none', class_weight='balanced',
                            solver='saga', max_iter=100000)
    rf = RandomForestClassifier(class_weight='balanced_subsample')
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

    sorted_coefs = sorted(coefs, key=abs)
    fig, axes = plt.subplots(1, 2)
    axes[0].barh(list(range(len(coefs))), sorted_coefs)
    axes[1].barh(list(range(len(features))), sorted(features))
    plt.show()
    
    

    