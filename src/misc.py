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
from model_class import Model, create_model
from plots import feature_comparison, plot_learning_curve
from xgboost import XGBClassifier
# from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
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
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,
    #                                                     stratify=y,
    #                                                     random_state=42)
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


if __name__ == '__main__':
    all_df = create_data('../data/merged_df_test.csv', 'All')
    # all_df.full_df['random'] = np.random.random(len(all_df.full_df))
    # print(variance_factor(all_df.full_df))

    # compare best models from random search
    X, y = Data.pop_reported(all_df.full_df)
    all_df.train_test_split(test_size=0.33)
    X_train, y_train = all_df.pop_reported(all_df.train_df)
    X_test, y_test = all_df.pop_reported(all_df.test_df)
    lr = LogisticRegression(penalty='none', class_weight='balanced',
                            solver='saga', max_iter=100000)
    rf = RandomForestClassifier(class_weight='balanced_subsample')
    xgb = XGBClassifier()
    grad_boost = XGBClassifier(learning_rate =0.01,
                               n_estimators=551,
                               max_depth=6,
                               min_child_weight=0,
                               gamma=0.275,
                               subsample=0.6,
                               colsample_bytree=0.5,
                               objective= 'binary:logistic',
                               nthread=4,
                               scale_pos_weight=1,
                               reg_alpha=0.2,
                               seed=27)
    # nn_model = KerasClassifier(build_fn=create_model, batch_size=100, epochs=50)
    mod_list = [(lr, 'Logistic Regression', 'blue'),
                (rf, 'Random Forest', 'orange'),
                (xgb, 'XGBoost Classifier', 'purple'),
                (grad_boost, 'XG Boost', 'purple')]
                # (nn_model, 'Neural Net')]
    scores, model_list, mod_class_list = compare_models(
        mod_list[:-4:-2], f1_score, X_train, X_test, y_train, y_test,
        fig_name='../images/boost_rand_comp.png', save=False
    )
    print(scores)

    '''
    # plot feature importance and coefs
    log_complex = model_list[0][0]
    random_forest = model_list[1][0]
    coefs = log_complex.coef_[0]
    features = random_forest.feature_importances_
    columns = all_df.full_df.drop('reported', axis=1).columns
    feature_comparison(columns, coefs, features, save=True)
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

    # Evaluate XGBoost model
    # Confusion Matrix (Bar Chart)
    tp, fp, fn, tn = mod_class_list[0].confusion_matrix(X_test, y_test.values)
    plt.bar([1,2], [fp, 0], color=['r', 'g'], alpha=0.5,
            label='Not Reported')
    plt.bar([1,2], [0, fn], color=['g', 'g'], alpha=0.5,
            label='Reported')
    plt.xlabel('Predicted Results')
    plt.ylabel('Incorrect Classifications')
    plt.xticks([1, 2], ['Reported', 'Not Reported'])
    plt.legend(title='Actual Result')
    plt.title('Less Confusion')

    # Learning Curve
    # plot_learning_curve(boosted_forest, 'Learning Curve', X, y)
    plt.show()

    # List of incorrectly classified chromatograms
    y_test_df = pd.DataFrame(y_test)
    y_test_df['probas'] = mod_class_list[0].predict_proba(X_test)[:, 1]
    new_df = pd.merge(X_test, y_test_df, how='left', left_index=True,
                      right_index=True)
    new_df['preds'] = np.where(new_df['probas']
                               < mod_class_list[0].best_thresh, 0, 1)
    new_df['fp'] = np.where(
        new_df['preds'] == 1, np.where(new_df['reported'] == 0, 1, 0), 0
    )
    new_df['fn'] = np.where(
        new_df['preds'] == 0, np.where(new_df['reported'] == 1, 1, 0), 0
    )
    sample_df = pd.read_csv('../data/chrom_info.csv', index_col='Unnamed: 0')
    fp_index = list(new_df[new_df['fn'] == 1].index.values)
    print(sample_df[sample_df.index.isin(fp_index)])