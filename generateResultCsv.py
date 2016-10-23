# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.linear_model import LogisticRegression as LR
from sklearn import svm
from sklearn import cross_validation
from sklearn.preprocessing import StandardScaler
from tools_removeRedundancy import generateOfflineTestFeatures, processOfflineTest

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

if __name__ == '__main__':
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
    result.to_csv('result_svc_v3.0.csv', index=False, header=None)
    print y_predict[:20]