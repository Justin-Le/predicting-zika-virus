import numpy as np
import pandas as pd
from features_zika import *
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import log_loss, roc_auc_score, f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, \
                             ExtraTreesClassifier
import pdb

def main():
    data = pd.read_csv('./aegypti_albopictus.csv')

    # Remove uninformative features
    data = data.drop(['OCCURRENCE_ID', 'COUNTRY', 'COUNTRY_ID'], axis=1)

    # Encode
    categoricals = ['VECTOR', 'SOURCE_TYPE', 'LOCATION_TYPE', 
                    'POLYGON_ADMIN', 'GAUL_AD0', 'YEAR', 'STATUS']

    for c in categoricals:
        data[c] = pd.factorize(np.array(data[c]))[0]

    X = data.ix[:, 1:]
    y = data.ix[:, 0]

    # logistic_cv(X, y)
    # xgb_cv(X, y)

    seed = 1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    xgb_test(X_train, y_train, X_test, y_test)

    X, y = extract_features(X, y)
    
    # X = X.drop('GAUL_AD0', axis=1)
    print(X.head())

    # logistic_cv(X, y)
    # xgb_cv(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    xgb_test(X_train, y_train, X_test, y_test)

    """
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)

    print("="*80)
    print("\nLogistic regression performance on unseen data:")
    print("\nlog-loss: %.4f" % log_loss(y_test, pred))
    print("\nAUC: %.4f" % roc_auc_score(y_test, pred))
    print("\nF1 score: %.4f" % f1_score(y_test, pred))
    print("\nAccuracy: %.4f" % accuracy_score(y_test, pred))
    print("="*80)
    """

def logistic_cv(X, y):
    # Instantiate logistic regression
    clf = LogisticRegression()

    #Cross-validate logistic regression
    print("\nLogistic regression results for 10-fold cross-validation:\n")
    scores = cross_val_score(clf, X, y, cv=10, scoring='accuracy')
    print(("Accuracies:\n %s\n\n" +
           "Best accuracy on held-out data: %.4f\n\n" +
           "Mean accuracy on held-out data: %.4f\n\n") % (str(scores), scores.max(), scores.mean()))
    scores = cross_val_score(clf, X, y, cv=10, scoring='roc_auc')
    print(("AUC:\n %s\n\n" +
           "Best AUC on held-out data: %.4f\n\n" + 
           "Mean AUC on held-out data: %.4f\n\n") % (str(scores), scores.max(), scores.mean()))

def xgb_cv(X, y):
    # Instantiate XGBoost
    n_estimators = 150
    dtrain = xgb.DMatrix(X, y)

    # XGBoost was tuned on the raw data.
    bst = XGBClassifier(n_estimators=n_estimators, #70
                        max_depth=3, 
                        min_child_weight=5, 
                        gamma=0.5, 
                        learning_rate=0.05, 
                        subsample=0.7, 
                        colsample_bytree=0.7, 
                        reg_alpha=0.001,
                        seed=1,
                        silent=True)

    # Cross-validate XGBoost
    params = bst.get_xgb_params() # Extract parameters from XGB instance to be used for CV
    num_boost_round = bst.get_params()['n_estimators'] # XGB-CV has different names than sklearn

    cvresult = xgb.cv(params, dtrain, num_boost_round=num_boost_round, 
                      nfold=10, metrics=['logloss', 'auc', 'error'], seed=1)

    """
    print("="*80)
    print("\nXGBoost results for 10-fold cross-validation:")
    print(cvresult)
    print("="*80)
    """

    # XGBoost summary
    print("="*80)
    print("\nXGBoost summary for 100 rounds of 10-fold cross-validation:")
    print("\nBest mean log-loss: %.4f" % cvresult['test-logloss-mean'].min())
    print("\nBest mean AUC: %.4f" % cvresult['test-auc-mean'].max())
    print("\nBest mean error: %.4f" % cvresult['test-error-mean'].min()) 
    print("="*80)

def xgb_test(X_train, y_train, X_test, y_test):
    n_estimators = 150

    bst = XGBClassifier(n_estimators=n_estimators, #70
                        max_depth=3, 
                        min_child_weight=5, 
                        gamma=0.5, 
                        learning_rate=0.05, 
                        subsample=0.7, 
                        colsample_bytree=0.7, 
                        reg_alpha=0.001,
                        seed=1,
                        silent=True)

    bst.fit(X_train, y_train, eval_metric='logloss')
    pred = bst.predict(X_test)

    print("="*80)
    print("\nXGBoost performance on unseen data:")
    print("\nlog-loss: %.4f" % log_loss(y_test, pred))
    print("\nAUC: %.4f" % roc_auc_score(y_test, pred))
    print("\nF1 score: %.4f" % f1_score(y_test, pred))
    print("\nAccuracy: %.4f" % accuracy_score(y_test, pred))
    print("="*80)

    print(cross_val_score(bst, X_train, y_train, cv=10, scoring="roc_auc").mean())

if __name__ == "__main__":
    main()
