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
from plots import (feature_comparison, plot_learning_curve, confusion_plot,
                   profit_curve, stack_profit_curves)
from xgboost import XGBClassifier
from collections import defaultdict


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


def compare_models(model_list, metric, X_train, X_test, y_train, y_test,
                   thresh=0.5, fig_name='plot.png', save=False):
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
    fig_name: str
        file name for saving figure
    save: bool
        if true, save the plot
    Return
    ------
    metric_result: list
        list of tuples (model name, metric score)
    model_list: list
        original list of models
    mod_class_list:
        list of model class created from param model_list
    '''
    mod_class_list = []
    metric_result = []
    fig, axes = plt.subplots(1, 2, figsize=(10, 6))

    for model in model_list:
        mod = Model(model[0], metric)
        mod.fit(X_train, y_train)
        axes[0] = mod.roc_plot(X_test, y_test, axes[0], model[2], model[1])
        axes[1], thresh = mod.thresh_plot(X_test, y_test,
                                            axes[1], model[2], model[1])
        result = mod.score_metric(X_test, y_test, thresh)
        metric_result.append((model[1], result.round(2)))

        print(model[1])
        print(mod.summary(X_test, y_test))
        mod_class_list.append(mod)
    axes[0].plot([0, 1], [0, 1], color='navy', linestyle='--',
                 label='No Skill')
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('ROC Curve')
    axes[0].legend()
    axes[1].set_xlabel('Threshold')
    axes[1].set_ylabel('F1 Score')
    axes[1].set_title('Threshold Comparison')
    if save:
        plt.savefig(fig_name)
    else:
        plt.show()
    return metric_result, model_list, mod_class_list


def incorrect_classifications(X_test, y_test, model,
                              sample_file='../data/chrom_info.csv'):
    '''Create a list of incorrectly classified chromatograms
    Parameters
    ----------
    X_test: pandas DataFrame
        test dataframe of features
    y_test: pandas DataFrame
        test dataframe of targets
    model: Model class
        model from the model class to use for classifications
    sample_file: str
        location of the file with sample and analyte values
    Return
    ------
    fp_df: pandas DataFrame
        dataframe of sample and analyte name for false positives
    fn_df: pandas DataFrame
        dataframe of sample and analyte name for false negatives
    '''
    y_test_df = pd.DataFrame(y_test)
    y_test_df['probas'] = model.predict_proba(X_test)[:, 1]
    new_df = pd.merge(X_test, y_test_df, how='left', left_index=True,
                      right_index=True)
    new_df['preds'] = np.where(new_df['probas']
                               < model.best_thresh, 0, 1)
    new_df['fp'] = np.where(
        new_df['preds'] == 1, np.where(new_df['reported'] == 0, 1, 0), 0
    )
    new_df['fn'] = np.where(
        new_df['preds'] == 0, np.where(new_df['reported'] == 1, 1, 0), 0
    )
    sample_df = pd.read_csv(sample_file, index_col='Unnamed: 0')
    fp_index = list(new_df[new_df['fp'] == 1].index.values)
    fp_df = sample_df[sample_df.index.isin(fp_index)]
    fn_index = list(new_df[new_df['fn'] == 1].index.values)
    fn_df = sample_df[sample_df.index.isin(fn_index)]
    return fp_df, fn_df

if __name__ == '__main__':
    all_df = create_data('../data/merged_df_test.csv', 'All')

    # Create train/test sets
    X, y = Data.pop_reported(all_df.full_df)
    all_df.train_test_split(test_size=0.25)
    X_train, y_train = all_df.under_sampling(all_df.train_df)
    X_test, y_test = all_df.pop_reported(all_df.test_df)

    # Create models from best hyperparameter searches
    lr = LogisticRegression(penalty='none', class_weight='balanced',
                            solver='saga', max_iter=100000)
    rf = RandomForestClassifier(class_weight='balanced_subsample')
    xgb = XGBClassifier()
    grad_boost = XGBClassifier(learning_rate =0.01,
                               n_estimators=551,
                               max_depth=4,
                               min_child_weight=1.5,
                               gamma=0,
                               subsample=0.7,
                               colsample_bytree=0.5,
                               objective= 'binary:logistic',
                               nthread=4,
                               scale_pos_weight=2,
                               reg_lambda=0,
                               reg_alpha=0.005,
                               eta=0.125,
                               seed=27)

    # Compare best models
    mod_list = [(lr, 'Logistic Regression', 'blue'),
                (rf, 'Random Forest', 'orange'),
                # (xgb, 'XGBoost Classifier', 'purple'),
                (grad_boost, 'XG Boost', 'purple')]
    scores, model_list, mod_class_list = compare_models(
        mod_list, f1_score, X_train, X_test, y_train, y_test,
        fig_name='../images/boost_rand_comp.png', save=False
    )
    print(scores)
    '''
    # plot feature importance and coefs
    # log_reg = model_list[0][0]
    # log_tup = ('green', 'Logistic Regression', 'Coefficient Value')
    # coefs = log_reg.coef_[0]
    boosted_forest = model_list[0][0]
    boost_tup = ('purple', 'XG Boost', 'Feature Importance')
    features1 = boosted_forest.feature_importances_
    random_forest = model_list[1][0]
    rand_tup = ('orange', 'Random Forest', 'Feature Importance')
    features2 = random_forest.feature_importances_
    columns = all_df.full_df.drop('reported', axis=1).columns
    labels = [boost_tup, rand_tup]
    feature_comparison(columns, features1, features2, labels,
                       fig_name='../images/boost_rand_features.png',
                       save=False)
    '''
    # Evaluate XGBoost model
    # Confusion Matrix (Bar Chart)
    mod_boost = mod_class_list[2]
    cost_matrix = (0, 0, -1, 0.5)
    tp, fp, fn, tn = mod_boost.confusion_matrix(X_test, y_test.values, 0.6)
    confusion_plot(tp, fp, fn, tn, fig_name = '../images/bar_confusion.png',
                   save=False)
    print(tp, fp, fn, tn)

    # Learning Curve
    # plot_learning_curve(grad_boost, 'Learning Curve', X, y, axes=None,
    #                     ylim=None, cv=10, n_jobs=-1,
    #                     train_sizes=np.linspace(.1, 1.0, 50))
    # plt.savefig('../images/learning.png')

    # List of incorrectly classified chromatograms
    fp_df, fn_df = incorrect_classifications(X_test, y_test,
                                             mod_boost)

    # Profit Curves
    fig, ax = plt.subplots(1)
    profit_curve(mod_boost, X_test, y_test, cost_matrix, ax=ax, label='XGBoost',
                 fig_name='../images/profit.png', save=False)

    sampling_methods = [all_df.under_sampling(all_df.train_df),
                        all_df.over_sampling(all_df.train_df),
                        all_df.smote_sampling(all_df.train_df),
                        all_df.pop_reported(all_df.train_df)]
    label_sampling = ['Under', 'Over', 'SMOTE', 'None']
    label_models = ['Logistic', 'Random Forest', 'Tuned XGBoost']
    model_colors = ['blue', 'orange', 'purple']
    stack_profit_curves(mod_class_list, sampling_methods, mod_class_list[2],
                        label_models, label_sampling, X_test, y_test,
                        cost_matrix, model_colors,
                        fig_name='../images/profit_curve', save=False)
   

    X, y = all_df.under_sampling(all_df.full_df)
    plot_learning_curve(mod_boost.model, 'Learning Curve', X, y, axes=None,
                        ylim=None, cv=10, n_jobs=-1,
                        train_sizes=np.linspace(.1, 1.0, 10))
    plt.savefig('../images/learning.png')