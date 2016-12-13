# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
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
from sklearn import preprocessing

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

# 随机森林分类
def randomForestClassifierTest(x, y):
    pass

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
    return pd.DataFrame(y_predict), model

# svm-svc
def svcTest(x, y, x_predict):
    model = svm.SVC(kernel='linear', probability=True)
    model.fit(x, y)
    y_predict = model.predict_proba(x_predict)
    return pd.DataFrame(y_predict), model

#标准化处理
def processStandard(x, x_predict):
    scaler = preprocessing.StandardScaler().fit(x)
    print scaler.mean_
    print scaler.std_
    x = scaler.transform(x)
    x_predict = scaler.transform(x_predict)
    return x, x_predict

#最小最大缩放
def processMinMaxScaler(x, x_predict):
    scalar = preprocessing.MinMaxScaler().fit(x)
    print scalar.min_
    print scalar.scale_
    x = scalar.transform(x)
    x_predict = scalar.transform(x_predict)
    return x, x_predict

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
    train.drop(['Unnamed: 0'], axis=1, inplace=True)

    temp = train[train.result==1].copy()
    for i in xrange(10):
        train = pd.concat([train, temp])

    train.index = np.arange(len(train))
    rei = np.random.permutation(len(train))
    train = train.ix[rei]

    x = train[["Distance","userRate","merchantRate","discountRate", 'week']]
    y = train[['result']]
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)

    y_predict_test, model = logisticRegressionTest(x_train, y_train, x_test)
    print 'auc: ', get_auc(y_test, pd.DataFrame(y_predict_test)[1])
    #
    test = pd.read_csv('offlineTestfeaturesDistanceRandom.csv')
    x_predict = test[["Distance","userRate","merchantRate","discountRate", 'week']]
    y_predict = model.predict_proba(x_predict.values)

    offlineTest = pd.read_csv('offlineTestWithCost.csv')
    offlineTest = offlineTest[['User_id', 'Coupon_id', 'Date_received']]
    offlineTest['result'] = pd.DataFrame(y_predict)[1]

    offlineTest.to_csv('result_v13.0.csv', index=False, header=None)

# features：“Distance","userRate","merchantRate","discountRate", 'week', 'userTotalCost', 'userVaildedCost', 'merchantCost', 'merchantVaildedCost'
def testFeaturesCost():
    train = pd.read_csv('offlineTrainWithCost.csv')
    x = train[["Distance","userRate","merchantRate","discountRate", 'week', 'userTotalCost', 'userVaildedCost', 'merchantCost', 'merchantVaildedCost']]
    y = train['result'].astype(int)

    test = pd.read_csv('offlineTestWithCost.csv')
    x_predict = test[["Distance","userRate","merchantRate","discountRate", 'week', 'userTotalCost', 'userVaildedCost', 'merchantCost', 'merchantVaildedCost']]

    x, x_predict = processMinMaxScaler(x, x_predict)
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)
    y_predict_test, model = logisticRegressionTest(x_train, y_train, x_test)
    print 'auc: ', get_auc(y_test, y_predict_test[1])

    print x[:1]
    print x_predict[:1]
    y_predict = pd.DataFrame(model.predict_proba(x_predict))[1]
    offlineTest = test[['User_id', 'Coupon_id', 'Date_received']]
    offlineTest['result'] = y_predict.map(lambda x: round(x, 2))
    #
    offlineTest.to_csv('result_with_log_round_v12.0.csv', index=False, header=None)

def format_zcs(x):
    #归一化
    #1.Distance、week
    x['Distance']=x['Distance']/6.0
    mask = x['Distance']==0.0
    x['Distance'][mask] = 0.000001
    x['week']=x['week']/6.0
    mask = x['week']==0.0
    x['week'][mask] = 0.000001
    #2.userRate、merchantRate
    mask = x['userRate']==0.0
    x['userRate'][mask] = 0.000001
    mask = x['merchantRate']==0.0
    x['merchantRate'][mask] = 0.000001
    #3.userTotalCost、userVaildedCost、merchantCost、merchantVaildedCost
    cur_max=x['userTotalCost'].max()
    if cur_max==0:cur_max=1
    x['userTotalCost']=x['userTotalCost']/cur_max
    cur_max=x['userVaildedCost'].max()
    if cur_max==0:cur_max=1
    x['userVaildedCost']=x['userVaildedCost']/cur_max
    cur_max=x['merchantCost'].max()
    if cur_max==0:cur_max=1
    x['merchantCost']=x['merchantCost']/cur_max
    cur_max=x['merchantVaildedCost'].max()
    if cur_max==0:cur_max=1
    x['merchantVaildedCost']=x['merchantVaildedCost']/cur_max
    #4.
    mask = x['userTotalCost']==0.0
    x['userTotalCost'][mask] = 0.000001
    mask = x['userVaildedCost']==0.0
    x['userVaildedCost'][mask] = 0.000001
    mask = x['merchantCost']==0.0
    x['merchantCost'][mask] = 0.000001
    mask = x['merchantVaildedCost']==0.0
    x['merchantVaildedCost'][mask] = 0.000001
    return x

# features：“Distance","userRate","merchantRate","discountRate", 'week', 'userTotalCost', 'userVaildedCost', 'merchantCost', 'merchantVaildedCost'
def testFeaturesCost_zcs():
    train = pd.read_csv('offlineTrainWithCost.csv')
    x = train[["Distance","userRate","merchantRate","discountRate", 'week', 'userTotalCost', 'userVaildedCost', 'merchantCost', 'merchantVaildedCost']]
    y = train['result'].astype(int)
    x=format_zcs(x)
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)
    y_predict_test, model = logisticRegressionTest(x_train, y_train, x_test)
    print 'auc: ', get_auc(y_test, y_predict_test[1])

    test = pd.read_csv('offlineTestWithCost.csv')
    x_predict = test[["Distance","userRate","merchantRate","discountRate", 'week', 'userTotalCost', 'userVaildedCost', 'merchantCost', 'merchantVaildedCost']]
    y_predict = pd.DataFrame(model.predict_proba(x_predict.values))[1]

    offlineTest = test[['User_id', 'Coupon_id', 'Date_received']]
    offlineTest['result'] = y_predict

    offlineTest.to_csv('result_with_cost_v6.0.csv', index=False, header=None)

testFeatures_DistanceRandom()