import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score, f1_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from data_class import Data
import statsmodels.api as sm

class Model(object):

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
        return ax


if __name__ == '__main__':
    cols_drop_list = ['Analyte Start Scan', 'Analyte Stop Scan',
                        'Analyte Centroid Location (min)',
                        # 'Relative Retention Time',
                        # 'Analyte Integration Quality',
                        # 'Analyte Peak Area (counts)',
                        # 'Analyte Peak Height (cps)',
                        'Analyte Peak Width (min)',
                        # 'Analyte Peak Width at 50% Height (min)',
                        'height_ratio',
                        'area_ratio',
                        # 'Analyte Start Time (min)',
                        'Analyte Stop Time (min)']

    all_df = Data('../data/merged_df.csv', 'All', cols_drop_list)
    # print(variance_factor(all_df.limited_df).to_markdown())

