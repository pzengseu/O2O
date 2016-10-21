# -*- coding: utf-8 -*-
import pandas as pd
import time

'''
计算线下训练集
1、将日期转化为星期
2、计算用户消费优惠券间隔天数
3、计算每条数据是否在规定日期内使用优惠券，1为正样本，0为负样本
'''
def processOfflineTrain(file):
    offline = pd.read_csv(file, header=None)
    offline.columns = ['User_id', 'Merchant_id', 'Coupon_id', 'Discount_rate', 'Distance', 'Date_received', 'Date']
    offline = offline[offline['Coupon_id'] != 'null']
    offline['result'] = [0] * len(offline)

    #将日期转化为星期
    offline['week'] = offline['Date_received'].apply(lambda x: time.strptime(str(x), '%Y%m%d')[6])

    #计算用户消费优惠券间隔天数
    cost = offline[offline['Date'] != 'null']
    offline['days'] = cost['Date'].apply(lambda x: int(time.strptime(str(x), '%Y%m%d')[7])) - \
                      cost['Date_received'].apply(lambda x: int(time.strptime(str(x), '%Y%m%d')[7]))
    offline['days'] = offline['days'].fillna(1000)

    #计算每条数据是否在规定日期内使用优惠券，1为正样本，0为负样本
    offline['result'] = offline[offline['days'] <= 15]['days'].apply(lambda x: 1)
    offline['result'] = offline['result'].fillna(0)

    #Distance为null,设为11
    offline['Distance'] = offline['Distance'].map(lambda x:11 if x=='null' else x)
    #不成熟的想法，运算速度奇慢
    # for i in xrange(len(offline)):
    #     if offline['Date_received'][i] != 'null':
    #         receivedDay = time.strptime(offline['Date_received'][i], '%Y%m%d')
    #         offline['week'][i] = receivedDay[6]
    #         if offline['Date'][i] != 'null':
    #             dateDay = time.strptime(offline['Date'][i], '%Y%m%d')
    #             offline['days'][i] = int(dateDay[7])-int(receivedDay[7])
    #             if offline['days'][i] <= 15: offline['result'][i] = 1
    #             print i, offline['days'][i]

    offline[['week', 'days', 'result', 'Distance']] = offline[['week', 'days', 'result', 'Distance']].astype(int)
    # print offline.loc[:100, ['Date_received', 'Date', 'week', 'days', 'result']]
    return offline

#处理线下测试集
def processOfflineTest(file):
    offlineTest = pd.read_csv(file, header=None)
    offlineTest.columns = ['User_id', 'Merchant_id', 'Coupon_id', 'Discount_rate', 'Distance', 'Date_received']
    offlineTest['week'] = offlineTest['Date_received'].map(lambda x: time.strptime(str(x), '%Y%m%d')[6]).astype(int)
    offlineTest['week'] = offlineTest['week'].fillna(7) #7代表日期为空
    return offlineTest

#用户ID使用优惠券频率字典
def processOfflineUserID(offline):
    userIdDict = {}
    userId = set(offline['User_id'])
    for u in userId:
        # print u
        rate = 0
        userIdDict[u] = 0
        user = offline[offline['User_id'] == u]
        totalReceived = len(user)
        totalCostDate = len(user[user['Date'] != 'null'])
        if totalReceived > 0 : rate = float(totalCostDate) / totalReceived
        userIdDict[u] = [totalReceived, totalCostDate, rate]

    f = file('userId.txt', 'w+')
    for u in userIdDict:
        f.write(str(u)+' '+' '.join(map(str, userIdDict[u]))+'\n')
    f.close()
    return userIdDict

#商户ID使用优惠券频率字典
def processOfflineMerchantID(offline):
    merchantIdDict = {}
    merchantId = set(offline['Merchant_id'])
    for m in merchantId:
        # print m
        rate = 0
        merchantIdDict[m] = 0
        merchant = offline[offline['Merchant_id'] == m]
        totalSended = len(merchant)
        totalCost = len(merchant[merchant['Date'] != 'null'])
        if totalSended > 0: rate = float(totalCost) / totalSended
        merchantIdDict[m] = [totalSended, totalCost, rate]

    f = file('merchantId.txt', 'w+')
    for m in merchantIdDict:
        f.write(str(m)+' '+' '.join(map(str, merchantIdDict[m]))+'\n')
    f.close()

    return merchantIdDict

#折扣率频率
def processOfflineDiscountRate(offline):
    discountDict = {}
    discountRate = set(offline['Discount_rate'])
    for d in discountRate:
        discountDict[d] = 0
        discount = offline[offline['Discount_rate'] == d]
        totalSended = len(discount)
        totalCosted = len(discount[discount['Date'] != 'null'])
        rate = float(totalCosted) / totalSended

        discountDict[d] = [totalSended, totalCosted, rate]

    f = file('discount.txt', 'w+')
    for d in discountDict:
        f.write(str(d)+' '+' '.join(map(str, discountDict[d]))+'\n')
    f.close()

    return discountDict

#获取特征并保存到文件
def generateFeatures(offline):
    userDict = {}
    merchantDict = {}
    discountDict = {}

    ufile = file('userId.txt', 'r')
    for line in ufile:
        d = line.split()
        userDict[d[0]] = d[3]
    ufile.close()

    mfile = file('merchantId.txt', 'r')
    for line in mfile:
        d = line.split()
        merchantDict[d[0]] = d[3]
    mfile.close()

    dfile = file('discount.txt', 'r')
    for line in dfile:
        d = line.split()
        discountDict[d[0]] = d[3]
    dfile.close()

    offline['userRate'] = offline['User_id'].map(lambda x: userDict[str(x)])
    offline['merchantRate'] = offline['Merchant_id'].map(lambda x: merchantDict[str(x)])
    offline['discountRate'] = offline['Discount_rate'].map(lambda x: discountDict[str(x)])

    features = offline.drop(['User_id', 'Merchant_id', 'Coupon_id', 'Discount_rate', 'Date_received', 'Date', 'days'],
                         axis=1)
    features[['userRate', 'merchantRate', 'discountRate']] = features[['userRate', 'merchantRate', 'discountRate']].astype(float)

    features.to_csv('offlineTrainfeatures.csv')
    return features


offline = processOfflineTrain('ccf_offline_stage1_train.csv')
# processOfflineUserID(offline)
# processOfflineDiscountRate(offline)
# processOfflineMerchantID(offline)