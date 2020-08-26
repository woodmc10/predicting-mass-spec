import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score, f1_score, roc_curve, make_scorer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from data_class import Data
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
                                  ('log', self.model)])

    def fit(self, X, y):
        self.pipeline.fit(X, y)
    
    def log_coef_(self):
        return self.model.coef_

    def predict(self, X):
        return self.pipeline.predict(X)

    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)

    def score_metric(self, X, y, thresh=0.5):
        y_prob = self.predict_proba(X)[:, 1]
        y_pred = (y_prob >= thresh).astype(int)
        return self.metric(y, y_pred)

    def roc(self, X, y):
        y_probs = self.predict_proba(X)[:, 1]
        fpr, tpr, thresholds = roc_curve(y, y_probs)
        return fpr, tpr
    
    def roc_plot(self, X, y, ax, label):
        fpr, tpr = self.roc(X, y)
        ax.plot(fpr, tpr, label=label)
        return ax

    def thresh_plot(self, X, y, ax, label):
        y_probs = self.predict_proba(X)[:, 1]
        thresholds = np.linspace(0, 1, 51)
        metrics = [self.score_metric(X, y, thresh) for thresh in thresholds]
        ax.plot(thresholds, metrics, label=label)
        best_thresh = thresholds[np.argmax(metrics)]
        return ax, best_thresh


if __name__ == '__main__':
    cols_drop_list = ['Analyte Start Scan', 'Analyte Stop Scan',
                        'Analyte Centroid Location (min)',
                        'Relative Retention Time',
                        'Analyte Integration Quality',
                        # 'Analyte Peak Area (counts)',
                        'Analyte Peak Height (cps)',
                        # 'Analyte Peak Width (min)',
                        'Analyte Peak Width at 50% Height (min)',
                        'height_ratio',
                        'area_ratio',
                        'Analyte Start Time (min)',
                        'Analyte Stop Time (min)']

    all_df = Data('../data/merged_df.csv', 'All', cols_drop_list)
    # print(variance_factor(all_df.limited_df).to_markdown())
    X, y = Data.pop_reported(all_df.limited_df)

    logistic = LogisticRegression(solver='saga', class_weight='balanced')
    distributions = dict(C=uniform(loc=0, scale=4),
                         penalty=['l2', 'l1', 'elasticnet', 'none'],
                         l1_ratio=uniform())
    scorer = make_scorer(recall_score)
    clf = RandomizedSearchCV(logistic, distributions, n_iter=100, verbose=0,
                             scoring=scorer, n_jobs=-1, random_state=0)
    search = clf.fit(X, y)
    print(search.best_params_, '\n---\n---\n---')
    # {'C': 2.195254015709299, 'l1_ratio': 0.7151893663724195, 'penalty': 'none'} -- limited
    #{'C': 2.195254015709299, 'l1_ratio': 0.7151893663724195, 'penalty': 'none'} -- full

    random_forest = RandomForestClassifier(class_weight='balanced', verbose=0)
    distributions_rf = dict(n_estimators=[10, 50, 100, 200],
                         max_features=['auto', 'sqrt', 'log2', None],
                         min_samples_split=randint(2, 20),
                         min_samples_leaf=uniform(0, 0.5))
    clf_rf = RandomizedSearchCV(random_forest, distributions_rf, n_iter=100,
                             scoring=scorer, n_jobs=-1, random_state=0)
    search_rf = clf_rf.fit(X, y)
    print(search_rf.best_params_, '\n---\n---\n---')
    #{'max_features': None, 'min_samples_leaf': 0.3175294368017819, 'min_samples_split': 14, 'n_estimators': 10}  -- limited
    #{'max_features': None, 'min_samples_leaf': 0.3117550505659341, 'min_samples_split': 17, 'n_estimators': 50}  -- full