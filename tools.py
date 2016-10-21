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
        totalReceived = len(user[user['Date_received'] != 'null'])
        totalCostDate = len(user[(user['Date_received'] != 'null') & (user['Date'] != 'null')])
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
        totalSended = len(merchant[merchant['Date_received'] != 'null'])
        totalCost = len(merchant[(merchant['Date_received'] != 'null') & (merchant['Date'] != 'null')])
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


# offlineTest = processOfflineTest('ccf_offline_stage1_test_revised.csv')
offline = processOfflineTrain('ccf_offline_stage1_train.csv')
print len(offline[offline['Date_received'] != 'null'])
print len(offline[offline['Coupon_id'] != 'null'])
print len(offline[(offline['Coupon_id'] != 'null') & (offline['Date_received'] != 'null')])
processOfflineUserID(offline)
processOfflineDiscountRate(offline)
processOfflineMerchantID(offline)