# -*- utf-8 -*-
import pandas as pd

offlineTest = pd.read_csv('ccf_offline_stage1_test_revised.csv', header=None)
# offlineTest.columns = ['User_id', 'Merchant_id', 'Coupon_id', 'Discount_rate', 'Distance', 'Date_received']
print offlineTest.dtypes
# print offlineTest['Date_received'] > 0
# print len(set(offlineTest[0]))
# print len(set(offlineTest[1]))
print len(set(offlineTest[3]))
print '------------------'
offline = pd.read_csv('ccf_offline_stage1_train.csv', header=None)
# print len(offline)
# print len(set(offline[0]))
# print len(set(offline[1]))
print len(set(offline[3]))
# print '---------------------'
print len(set(offline[3]) & set(offlineTest[3]))
# print len(set(offline[1]) & set(offlineTest[1]))
# print len(set(offline[2]) & set(offlineTest[2]))
# print '--------------'
# online = pd.read_csv('ccf_online_stage1_train.csv', header=None)
# print len(online)
# print len(set(online[0]))
# print len(set(online[1]))
# print len(set(online[3]))
# print '--------------'
# print len(set(online[0]) & set(offline[0]))
# print len(set(online[1]) & set(offline[1]))
# print len(set(online[3]) & set(offline[2]))
# print len(online)