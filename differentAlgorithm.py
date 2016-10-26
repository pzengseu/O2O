# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import auc_score
from sklearn.cross_validation import train_test_split
from sklearn.metrics import make_scorer
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

scoring = make_scorer(accuracy_score, greater_is_better=True)

def get_model(estimator, parameters, X_train, y_train, scoring):
    model = GridSearchCV(estimator, param_grid=parameters, scoring=scoring)
    model.fit(X_train, y_train)
    return model.best_estimator_

def get_auc(y, y_pred_proba):
    score = auc_score(y, y_pred_proba)
    print score

def GridSearchfile(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)

    #Logistic regression
    # lg = LR(random_state=42)
    # parameters = {'C': [0.5,0.6,0.7,0.8,0.9,1.0], 'penalty':['l1', 'l2']}
    # lg_best = get_model(lg, parameters, x_train, y_train, scoring)
    # joblib.dump(lg_best, 'lg_weekInNumbers.model')

    #RandomForest
    # rfc = RandomForestClassifier(random_state=42, criterion='entropy', min_samples_split=5, oob_score=True)
    # parameters = {'n_estimators':[500], 'min_samples_leaf':[12]}
    # rfc_best = get_model(rfc, parameters, x_train, y_train, scoring)
    # joblib.dump(rfc_best, 'rfc_weekInNumbers.model')

    #SVC
    svc = svm.SVC(random_state=42, probability=True)
    paramters = {'kernel':['linear', 'poly', 'rbf']}
    svc_best = get_model(svc, paramters, x_train, y_train, scoring)
    joblib.dump(svc_best, 'svc/svc_weekInNumbers.model')

#逻辑斯提回归
def logisticRegressionTest(x, y, x_predict):
    model = LR()
    model.fit(x, y)
    y_predict = model.predict_proba(x_predict)

    return pd.DataFrame(y_predict), model

# 测试xgboost算法
def xgboostTest(x, y, x_predict):
    model = XGBClassifier(seed=42, max_depth=3, objective='binary:logistic', n_estimators=400)
    model.fit(x, y)
    print model.feature_importances_
    y_predict = model.predict_proba(x_predict)
    joblib.dump(model, 'xgb/xgb')
    return pd.DataFrame(y_predict), model

# 在特征为"Distance","userRate","merchantRate","discountRate", 'week_0', 'week_1', 'week_2', 'week_3', 'week_4', 'week_5', 'week_6'下测试
def testFeatures_weekInNumbers():
    trainData = pd.read_csv('offlineTrainfeatures_week_number.csv', header=0)
    x = trainData[["Distance","userRate","merchantRate","discountRate", 'week_0', 'week_1', 'week_2', 'week_3', 'week_4', 'week_5', 'week_6']]
    x[["Distance"]] = x[["Distance"]].astype(int)
    x[["userRate","merchantRate","discountRate"]] = x[["userRate","merchantRate","discountRate"]].astype(float)
    x[['week_0', 'week_1', 'week_2', 'week_3', 'week_4', 'week_5', 'week_6']] = x[['week_0', 'week_1', 'week_2', 'week_3', 'week_4', 'week_5', 'week_6']].astype(int)
    y = trainData[["result"]].astype(int)

    # features = pd.read_csv('offlineTestfeatures_week_number.csv', header=0)
    # features[["Distance"]] = features[["Distance"]].astype(int)
    # features[["userRate","merchantRate","discountRate"]] = features[["userRate","merchantRate","discountRate"]].astype(float)
    # features[['week_0', 'week_1', 'week_2', 'week_3', 'week_4', 'week_5', 'week_6']] = features[['week_0', 'week_1', 'week_2', 'week_3', 'week_4', 'week_5', 'week_6']].astype(int)
    #
    # x_predict = features[["Distance","userRate","merchantRate","discountRate", 'week_0', 'week_1', 'week_2', 'week_3', 'week_4', 'week_5', 'week_6']]

    # rfc = joblib.load('rfc/rfc_weekInNumbers.model')
    # print rfc.feature_importances_
    # print rfc.predict_proba(x_predict)[:10]
    # print get_auc(y.values, pd.DataFrame(rfc.predict_proba(x))[1].values)

# 特征为："Distance","userRate","merchantRate","discountRate", 'week',且Distance随机化
def testFeatures_DistanceRandom():
    train = pd.read_csv('offlineTrainfeaturesDistanceRandom.csv')
    x = train[["Distance","userRate","merchantRate","discountRate", 'week']]
    y = train[['result']]
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)

    y_predict_test, model = xgboostTest(x_train, y_train, x_test)
    print 'auc: ', get_auc(y_test, pd.DataFrame(y_predict_test)[1])

    test = pd.read_csv('offlineTestfeaturesDistanceRandom.csv')
    x_predict = test[["Distance","userRate","merchantRate","discountRate", 'week']]
    print model.predict_proba(x_predict.values)

# features：“Distance","userRate","merchantRate","discountRate", 'week', 'userTotalCost', 'userVaildedCost', 'merchantCost', 'userVaildedCost'
def testFeaturesCost():
    train = pd.read_csv('offlineTrainWithCost.csv')
    x = train[["Distance","userRate","merchantRate","discountRate", 'week', 'userTotalCost', 'userVaildedCost', 'merchantCost', 'userVaildedCost']]
    y = train['result'].astype(int)

    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)
    y_predict_test, model = logisticRegressionTest(x_train, y_train, x_test)
    print 'auc: ', get_auc(y_test, y_predict_test[1])

    test = pd.read_csv('offlineTestWithCost.csv')
    x_predict = test[["Distance","userRate","merchantRate","discountRate", 'week', 'userTotalCost', 'userVaildedCost', 'merchantCost', 'userVaildedCost']]
    y_predict = pd.DataFrame(model.predict_proba(x_predict.values))[1]

    offlineTest = test[['User_id', 'Coupon_id', 'Date_received']]
    offlineTest['result'] = y_predict

    offlineTest.to_csv('result_with_cost_v6.0.csv', index=False, header=None)

testFeaturesCost()