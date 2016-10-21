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
    offline['result'] = [0] * len(offline)

    #将日期转化为星期
    offline['week'] = offline[offline['Date_received'] != 'null']['Date_received'].apply(lambda x: time.strptime(str(x), '%Y%m%d')[6])
    offline['week'] = offline['week'].fillna(7) #7代表日期为空

    #计算用户消费优惠券间隔天数
    cost = offline[(offline['Date_received'] != 'null') & (offline['Date'] != 'null')]
    offline['days'] = cost['Date'].apply(lambda x: int(time.strptime(str(x), '%Y%m%d')[7])) - \
                      cost['Date_received'].apply(lambda x: int(time.strptime(str(x), '%Y%m%d')[7]))
    offline['days'] = offline['days'].fillna(1000)

    #计算每条数据是否在规定日期内使用优惠券，1为正样本，0为负样本
    offline['result'] = offline[offline['days'] <= 15]['days'].apply(lambda x: 1)
    offline['result'] = offline['result'].fillna(0)

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

    offline[['week', 'days', 'result']] = offline[['week', 'days', 'result']].astype(int)
    # print offline.loc[:100, ['Date_received', 'Date', 'week', 'days', 'result']]
    return offline

def processOfflineTest(file):
    offlineTest = pd.read_csv(file, header=None)
    offlineTest.columns = ['User_id', 'Merchant_id', 'Coupon_id', 'Discount_rate', 'Distance', 'Date_received']
    offlineTest['week'] = offlineTest['Date_received'].map(lambda x: time.strptime(str(x), '%Y%m%d')[6]).astype(int)
    offlineTest['week'] = offlineTest['week'].fillna(7) #7代表日期为空
    return offlineTest

#用户ID使用优惠券频率字典，并返回平均值
def processOfflineUserID(offline):
    userIdDict = {}
    userId = set(offline['User_id'])
    for u in userId:
        print u
        userIdDict[u] = 0
        user = offline[offline['User_id'] == u]
        totalReceived = user[user['Date_received'] != 'null'].size
        totalCostDate = user[(user['Date_received'] != 'null') & (user['Date'] != 'null')].size
        if totalReceived > 0 : userIdDict[u] = float(totalCostDate) / totalReceived

    userSeries = pd.Series(userIdDict, index=userIdDict.keys())

    f = file('userId.txt', 'w+')
    for u in userIdDict:
        f.write(str(u)+' '+str(userIdDict[u])+'\n')
    f.close()
    return userIdDict, userSeries.mean()

##商户ID使用优惠券频率字典，并返回平均值
def processOfflineMerchantID(offline):
    merchantIdDict = {}
    merchantId = set(offline['Merchant_id'])
    for m in merchantId:
        print m
        merchantIdDict[m] = 0
        merchant = offline[offline['Merchant_id'] == m]
        totalSended = merchant[merchant['Date_received'] != 'null'].size
        totalCost = merchant[(merchant['Date_received'] != 'null') & (merchant['Date'] != 'null')].size
        if totalSended > 0: merchantIdDict[m] = float(totalCost) / totalSended

    f = file('merchantId.txt', 'w+')
    for m in merchantIdDict:
        f.write(str(m)+' '+str(merchantIdDict[m])+'\n')
    f.close()
    merchantSeries = pd.Series(merchantIdDict, index=merchantIdDict.keys())
    return merchantIdDict, merchantSeries.mean()

def processOfflineDiscountRate(offline):
    discountDict = {}
    discountRate = set(offline['Discount_rate'])
    print len(discountRate)
    for d in discountRate:
        print d
        discountDict[d] = 0
        discount = offline[offline['Discount_rate'] == d]
        totalSended = discount.size
        totalCosted = discount[discount['Date'] != 'null'].size
        rate = float(totalCosted) / totalSended

        discountDict[d] = [totalSended, totalCosted, rate]

    f = file('discount.txt', 'w+')
    for d in discountDict:
        f.write(str(d)+' '+' '.join(map(str, discountDict[d]))+'\n')
    f.close()

    return discountDict

# offlineTest = processOfflineTest('ccf_offline_stage1_test_revised.csv')

offline = processOfflineTrain('ccf_offline_stage1_train.csv')
processOfflineDiscountRate(offline)