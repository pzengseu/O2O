# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.linear_model import LogisticRegression as LR
from sklearn import svm
from sklearn import cross_validation
from sklearn.metrics import auc_score
from sklearn.preprocessing import StandardScaler
from tools_removeRedundancy import generateOfflineTestFeatures, processOfflineTest, convertWeekInNumber

def get_auc(y, y_pred_proba):
    score = auc_score(y, y_pred_proba)
    print score

#逻辑斯提回归
def generatePredictWithLostic(x, y, x_predict):
    model = LR()

    # 交叉验证
    # cv = cross_validation.KFold(x.shape[0], 10, shuffle=True, random_state=33)
    # scores = cross_validation.cross_val_score(model, x, y, n_jobs=-1, cv=cv)
    # print scores

    # 数据标准缩放
    sc = StandardScaler()
    sc.fit(x)
    x_std = sc.transform(x)
    x_predict_std = sc.transform(x_predict)

    model.fit(x_std, y)
    y_proba = model.predict_proba(x_predict_std)
    y_proba = pd.DataFrame(y_proba)

    return y_proba

#linear svc
def generatePredictWithSVC(x, y, x_predict):
    model = svm.SVC(kernel='linear', probability=True)
    model.fit(x, y)
    y_predict = model.predict_proba(x_predict)

    return pd.DataFrame(y_predict)

def generateFileV1():
    trainData = pd.read_csv('offlineTrainfeatures.csv', header=0)
    x = trainData[["Distance","week","userRate","merchantRate","discountRate"]]
    y = trainData[['result']]
    x[["Distance","week"]] = x[["Distance","week"]].astype(int)
    x[["userRate","merchantRate","discountRate"]] = x[["userRate","merchantRate","discountRate"]].astype(float)
    y = trainData[["result"]].astype(int)

    offlineTest = processOfflineTest('ccf_offline_stage1_test_revised.csv')
    features = generateOfflineTestFeatures(offlineTest)
    features[["Distance","week"]] = features[["Distance","week"]].astype(int)
    features[["userRate","merchantRate","discountRate"]] = features[["userRate","merchantRate","discountRate"]].astype(float)

    x_predict = features[["Distance","week","userRate","merchantRate","discountRate"]]

    #Logistic
    # y_predict = generatePredictWithLostic(x.values, y.values, x_predict.values)

    #SVC
    y_predict = generatePredictWithSVC(x.values, y.values, x_predict.values)

    y_predict.to_csv('SVC_predict.csv')
    offlineTest['result'] = y_predict[1]
    result = offlineTest[['User_id', 'Coupon_id', 'Date_received', 'result']]
    # result.to_csv('result_svc_v3.0.csv', index=False, header=None)
    print y_predict[:20]

def generateFile_week_V2():
    trainData = pd.read_csv('offlineTrainfeatures_week_number.csv', header=0)
    x = trainData[["Distance","userRate","merchantRate","discountRate", 'week_0', 'week_1', 'week_2', 'week_3', 'week_4', 'week_5', 'week_6']]
    x[["Distance"]] = x[["Distance"]].astype(int)
    x[["userRate","merchantRate","discountRate"]] = x[["userRate","merchantRate","discountRate"]].astype(float)
    x[['week_0', 'week_1', 'week_2', 'week_3', 'week_4', 'week_5', 'week_6']] = x[['week_0', 'week_1', 'week_2', 'week_3', 'week_4', 'week_5', 'week_6']].astype(int)
    y = trainData[["result"]].astype(int)

    features = pd.read_csv('offlineTestfeatures_week_number.csv', header=0)
    features[["Distance"]] = features[["Distance"]].astype(int)
    features[["userRate","merchantRate","discountRate"]] = features[["userRate","merchantRate","discountRate"]].astype(float)
    features[['week_0', 'week_1', 'week_2', 'week_3', 'week_4', 'week_5', 'week_6']] = features[['week_0', 'week_1', 'week_2', 'week_3', 'week_4', 'week_5', 'week_6']].astype(int)

    x_predict = features[["Distance","userRate","merchantRate","discountRate", 'week_0', 'week_1', 'week_2', 'week_3', 'week_4', 'week_5', 'week_6']]

    #Logistic
    y_predict = generatePredictWithLostic(x.values, y.values, x_predict.values)

    #SVC
    # y_predict = generatePredictWithSVC(x.values, y.values, x_predict.values)

    offlineTest = processOfflineTest('ccf_offline_stage1_test_revised.csv')
    offlineTest['result'] = y_predict[1]
    result = offlineTest[['User_id', 'Coupon_id', 'Date_received', 'result']]
    result.to_csv('result_logistic_v4.0.csv', index=False, header=None)
    print y_predict[:20]

generateFile_week_V2()