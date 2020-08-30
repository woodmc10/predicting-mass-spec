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
                                 verbose=0, scoring=scorer, n_jobs=-1,
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
        return fpr, tpr

    def roc_plot(self, X, y, ax, label):
        '''Plot fpr vs tpr
        Parameters
        ----------
        X: numpy array
            array of features
        y: numpy array
            array of tagets
        ax: matplotlib axes
            axes for plotting curve
        label: string
            text label for curve

        Return
        ------
        ax: matplotlib axes
            axes with plotted roc curve
        '''
        fpr, tpr = self.roc(X, y)
        ax.plot(fpr, tpr, label=label)
        return ax

    def thresh_plot(self, X, y, ax, label):
        '''Plot threshold vs score of class metric
        Parameters
        ----------
        X: numpy array
            array of features
        y: numpy array
            array of tagets
        ax: matplotlib axes
            axes for plotting curve
        label: string
            text label for curve

        Return
        ------
        ax: matplotlib axes
            axes with plotted threshold curve
        best_thresh: float
            threshold that optimizes the model's metric
        '''
        y_probs = self.predict_proba(X)[:, 1]
        thresholds = np.linspace(0, 1, 51)
        metrics = [self.score_metric(X, y, thresh) for thresh in thresholds]
        ax.plot(thresholds, metrics, label=label)
        self.best_thresh = thresholds[np.argmax(metrics)]
        return ax, self.best_thresh


if __name__ == '__main__':
    all_df = create_data('../data/merged_df.csv', 'All')

    # logistic regression and random forest models - find best params
    X, y = Data.pop_reported(all_df.limited_df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,
                                                        stratify=y,
                                                        random_state=42)
    logistic = LogisticRegression(class_weight='balanced')
    Log = Model(logistic, f1_score)
    distributions = dict(C=uniform(loc=0, scale=4),
                         penalty=['l2', 'l1', 'elasticnet', 'none'],
                         l1_ratio=uniform())
    print(Log.hyper_search(distributions, X_train, y_train))
    random_forest = RandomForestClassifier(class_weight='balanced_subsample',
                                           random_state=43)
    RF = Model(random_forest, f1_score)
    distributions_rf = dict(n_estimators=[10, 50, 100, 200],
                            max_features=['auto', 'sqrt', 'log2', None],
                            min_samples_split=randint(2, 20),
                            min_samples_leaf=uniform(0, 0.5))
    print(RF.hyper_search(distributions_rf, X_train, y_train))
