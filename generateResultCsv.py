# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.linear_model import LogisticRegression as LR
from tools_removeRedundancy import generateOfflineTestFeatures, processOfflineTest

#逻辑斯提回归
def generatePredictWithLostic(x, y, x_predict):
    model = LR()
    model.fit(x, y)
    y_proba = model.predict_proba(x_predict)
    y = pd.DataFrame(y_proba)

    return y

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
    y_predict = generatePredictWithLostic(x.values, y.values, x_predict.values)

    offlineTest['result'] = y_predict[1]
    result = offlineTest[['User_id', 'Coupon_id', 'Date_received', 'result']]
    result.to_csv('result_logistic_v1.0.csv', index=False, header=None)

    print result[:10]