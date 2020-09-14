import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (recall_score, f1_score, roc_curve,
                             make_scorer, classification_report)
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from data_class import Data, create_data
import statsmodels.api as sm
from scipy.stats import uniform, randint
import xgboost as xgb
from xgboost.sklearn import XGBClassifier


class Model(object):
    '''class to store the pipeline and metric for creating and evaluting
    models, also contains code to create roc plots and plots of f1_score
    vs threshold for picking best threshold
    Parameters
    ----------
    model: sklearn model
    metric: sklearn metric
    '''
    def __init__(self, model, metric):
        self.model = model
        self.metric = metric
        self.pipeline = Pipeline([('scaler', StandardScaler()),
                                  ('model', self.model)])
        self.best_thresh = None

    def fit(self, X, y):
        self.pipeline.fit(X, y)

    def log_coef_(self):
        return self.model.coef_

    def predict(self, X):
        return self.pipeline.predict(X)

    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)

    def hyper_search(self, search_dict, X, y):
        '''Randomized Search CV
        Parameters
        ----------
        search_dict: dict
            dictionary of search terms and values for the model's
            hyperparameters
        X: numpy array
            array of features for training
        y: numpy array
            array of targets

        Return
        ------
        best_params_: dict
            dictionary of the parameters that had the best score
        '''
        scorer = make_scorer(self.metric)
        clf = RandomizedSearchCV(self.model, search_dict, n_iter=100,
                                 verbose=1, scoring=scorer, n_jobs=-1,
                                 random_state=32)
        search = clf.fit(X, y)
        return search.best_params_, search.best_score_

    def score_metric(self, X, y, thresh=0.5):
        '''Determine the score of the model based on the model's metric
        Parameters
        ----------
        X: numpy array
            array of features
        y: numpy array
            array of targets
        thresh: float
            threshold value between 0 and 1 to use for classification

        Return
        ------
        score from the metric
        '''
        y_prob = self.predict_proba(X)[:, 1]
        y_pred = (y_prob >= thresh).astype(int)
        return self.metric(y, y_pred)

    def summary(self, X, y, thresh=None):
        '''Create model summary from classification report
        Parameters
        ----------
        X: numpy array
            array of features
        y: numpy array
            array of targets
        thresh: float
            threshold to use for predictions, leave as none if
            best threshold has been determined for model

        Return
        ------
        report_df: DataFrame
            classification report as pandas dataframe
        '''
        if thresh is None:
            if self.best_thresh is None:
                thresh = 0.5
            else:
                thresh = self.best_thresh
        y_prob = self.predict_proba(X)[:, 1]
        y_pred = (y_prob >= thresh).astype(int)
        report = classification_report(y, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).T
        return report_df

    def roc(self, X, y):
        '''Create array of false positive rate and true positive rate for
        every threshold returned from predict_proba
        Parameters
        ----------
        X: numpy array
            array of features
        y: numpy array
            array of targets

        Return
        ------
        fpr: numpy array
            array of false positive rates calculated at each threshold
        tpr: numpy array
            array of true positive rates calculated at each threshold
        '''
        y_probs = self.predict_proba(X)[:, 1]
        fpr, tpr, thresholds = roc_curve(y, y_probs)
        return fpr, tpr, thresholds

    def roc_plot(self, X, y, ax, color, label):
        '''Plot fpr vs tpr
        Parameters
        ----------
        X: numpy array
            array of features
        y: numpy array
            array of tagets
        ax: matplotlib axes
            axes for plotting curve
        color: str
            color for curve
        label: string
            text label for curve

        Return
        ------
        ax: matplotlib axes
            axes with plotted roc curve
        '''
        fpr, tpr, _ = self.roc(X, y)
        ax.plot(fpr, tpr, color=color, label=label)
        return ax

    def thresh_plot(self, X, y, ax, color, label):
        '''Plot threshold vs score of class metric
        Parameters
        ----------
        X: numpy array
            array of features
        y: numpy array
            array of tagets
        ax: matplotlib axes
            axes for plotting curve
        color: str
            color for curve
        label: string
            text label for curve

        Return
        ------
        ax: matplotlib axes
            axes with plotted threshold curve
        best_thresh: float
            threshold that optimizes the model's metric
        '''
        thresholds = np.linspace(0, 1, 51)
        metrics = [self.score_metric(X, y, thresh) for thresh in thresholds]
        ax.plot(thresholds, metrics, color=color, label=label)
        self.best_thresh = thresholds[np.argmax(metrics)]
        return ax, self.best_thresh

    def confusion_matrix(self, X, y):
        '''Calculate fp, fn, tp, tn for a given dataset
        Params
        ------
        X: numpy array
            array of features
        y: numpy array
            array of tagets
        Return:
        -------
        fp, fn, tp, tn: int
            results evaluated at the best threshold if present
        '''
        if self.best_thresh is None:
            thresh = 0.5
        else:
            thresh = self.best_thresh
        y_prob = self.predict_proba(X)[:, 1]
        y_pred = (y_prob >= thresh).astype(int)
        fp = 0
        fn = 0
        tp = 0
        tn = 0
        for index, pred in enumerate(y_pred):
            if pred == 1:
                if y[index] == 1:
                    tp +=1
                else:
                    fp += 1
            else:
                if y[index] == 0:
                    tn += 1
                else:
                    fn += 1
        return tp, fp, fn, tn


if __name__ == '__main__':
    all_df = create_data('../data/merged_df.csv', 'All')

    # Create train/test sets
    X, y = Data.pop_reported(all_df.full_df)
    all_df.train_test_split(test_size=0.33)
    X_train, y_train = all_df.pop_reported(all_df.train_df)
    X_test, y_test = all_df.pop_reported(all_df.test_df)

    # Find best hyperparameters
    logistic = LogisticRegression(class_weight='balanced')
    Log = Model(logistic, f1_score)
    distributions = dict(C=uniform(loc=0, scale=4),
                         penalty=['l2', 'l1', 'elasticnet', 'none'],
                         l1_ratio=uniform())
    print(Log.hyper_search(distributions, X_train, y_train))
    # ({'C': 0.5051386999960084, 'l1_ratio': 0.02384165985593545,
    # 'penalty': 'l2'},
    # F1 Score: 0.4851561026625747)

    random_forest = RandomForestClassifier(class_weight='balanced_subsample',
                                           random_state=43)
    RF = Model(random_forest, f1_score)
    distributions_rf = dict(n_estimators=[10, 50, 100, 200],
                            max_features=['auto', 'sqrt', 'log2', None],
                            min_samples_split=randint(2, 20),
                            min_samples_leaf=uniform(0, 0.5))
    print(RF.hyper_search(distributions_rf, X_train, y_train))
    # ({'max_features': 'auto', 'min_samples_leaf': 0.004338663070259263,
    # 'min_samples_split': 7, 'n_estimators': 200},
    # F1 Score: 0.7767690571695022)

    grad_boost = XGBClassifier(random_state=43)
    GB = Model(grad_boost, f1_score)
    distributions_gb = dict(eta=[0.01, 0.025, 0.05, 0.075, 0.1, 0.125,
                                   0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3],
                            min_child_weight=[0.5, 1, 1.5, 2, 3, 5],
                            max_depth=[3, 4, 5, 6, 7, 8, 9, 10],
                            gamma=[0, 0.005, 0.01, 0.015, 0.02],
                            subsample=[0.5, 0.6, 0.7, 0.8, 0.9, 1],
                            # colsample_by_tree=[0.5, 0.6, 0.7, 0.8, 0.9, 1],
                            reg_lambda=[0, 1e-5, 0.001, 0.005, 0.01, 0.05,
                                        0.1, 1, 10, 100],
                            reg_alpha=[0, 1e-5, 0.001, 0.005, 0.01, 0.05,
                                        0.1, 1, 10, 100],
                            scale_pos_weight=[0, 0.5, 1, 2, 3],
                            objective=['binary:logistic', 'reg:linear',
                                       'multi:softprob'])
    print(GB.hyper_search(distributions_gb, X_train, y_train))
    # ({'subsample': 0.6, 'scale_pos_weight': 1, 'reg_lambda': 0.005,
    # 'reg_alpha': 0.01, 'objective': 'reg:linear', 'min_child_weight': 5,
    # 'max_depth': 5, 'gamma': 0, 'eta': 0.075, 'colsample_by_tree': 1},
    # F1 Score: 0.7961882282491608)
    
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
    xgtrain = xgb.DMatrix(X_train, y_train)
    cv_results = xgb.cv(grad_boost.get_xgb_params(), xgtrain,
                        metrics='aucpr', num_boost_round=5000,
                        early_stopping_rounds=50)
    # n_estimators=50
    distributions_gb = dict(reg_alpha=[0.005, 0.01, 0.02, 0.03, 0.04, 0.05,
                                       0.06, 0.07, 0.08, 0.09, 0.1, 0.2])
    # max_depth = 6
    # min_child_weight = 0
    # gamma = 0.275
    # subsample = 0.6
    # colsample_bytree = 0.5
    # reg_alpha = 0.01
    GB = Model(grad_boost, f1_score)
    print(GB.hyper_search(distributions_gb, X_train, y_train))
    print(cv_results)
    print(cv_results.shape)
