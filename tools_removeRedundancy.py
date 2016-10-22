# -*- coding: utf-8 -*-
import pandas as pd
import time
import numpy
import matplotlib.pyplot as plt

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

    offlineTest['Distance'] = offlineTest['Distance'].map(lambda x: 11 if x=='null' else x)
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

#获取训练集特征并保存到文件
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

#生成线下特征数据集
def generateOfflineTestFeatures(offlineTest):
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

    userMean = sum(map(float, userDict.itervalues())) / len(userDict)
    merchantMean = sum(map(float, merchantDict.itervalues())) / len(merchantDict)
    discountMean = sum(map(float, discountDict.itervalues())) / len(discountDict)

    offlineTest['userRate'] = offlineTest['User_id'].map(lambda x: userDict[str(x)] if str(x) in userDict else str(userMean))
    offlineTest['merchantRate'] = offlineTest['Merchant_id'].map(lambda x: merchantDict[str(x)] if str(x) in merchantDict else str(merchantMean))
    offlineTest['discountRate'] = offlineTest['Discount_rate'].map(lambda x: discountDict[str(x)] if str(x) in discountDict else str(discountMean))

    features = offlineTest.drop(['User_id', 'Merchant_id', 'Coupon_id', 'Discount_rate', 'Date_received'], axis=1)
    features[['userRate', 'merchantRate', 'discountRate']] = features[['userRate', 'merchantRate', 'discountRate']].astype(float)
    return features

def choose_testdata(x,y,num,random_state):
    """
    
        从训练数据集中选择部分数据作为测试数据，其余部分为训练数据
        :param x:数据集自变量部分
        :param y:数据集标签（因变量）部分
        :param num:将数据集划分成num份
        :param random_state:随机种子范围（状态值）
        :returns x_train:训练数据集自变量部分
        :returns y_train:训练数据集因变量部分
        :returns x_test:测试数据集自变量部分
        :returns y_est:测试数据集因变量部分
        
    """
    from sklearn.cross_validation import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1.0/num, random_state=random_state)
    x_test=pd.DataFrame(x_test)
    x_train=pd.DataFrame(x_train)
    y_train=pd.DataFrame(y_train)
    y_test=pd.DataFrame(y_test)
    y_train.columns = ['result']
    y_test.columns = ['result']
    return x_train, x_test, y_train, y_test


def proc_by_logis(x_train, x_test, y_train, y_test,threshold):
    """
    
        利用训练数据集训练逻辑斯蒂回归分类器，并对测试集进行预测
        :param x_train:训练数据集自变量部分
        :param y_train:训练数据集因变量部分
        :param x_test:测试数据集自变量部分
        :param y_est:测试数据集因变量部分
        :param threshold:逻辑斯蒂回归分类器的概率阈值
        :returns TPR:对测试集进行测试得到的tpr
        :returns FPR:对测试集进行测试得到的fpr
        :returns pred_y_probs1:对测试集进行测试得到的预测标签结果列表（取正例概率）
        
    """
    from sklearn.linear_model import LogisticRegression as LR
    lr=LR()
    lr.fit(x_train,y_train)
    #得到测试集预测结果
    pred_y=lr.predict_proba(x_test)
    pred_y=pd.DataFrame(pred_y)
    #正例概率列表
    pred_y_probs1=pred_y[1]
    pred_y_probs1=pd.DataFrame(pred_y_probs1)
    pred_y_probs1.columns = ['result']
    #求得TPR和FPR
    #print pred_y[:30]
    TP = sum((y_test['result']==1)& (pred_y_probs1['result']>=threshold))
    FN = sum((y_test['result']==1)& (pred_y_probs1['result']<threshold))
    FP = sum((y_test['result']==0)& (pred_y_probs1['result']>=threshold))
    TN = sum((y_test['result']==0)& (pred_y_probs1['result']<threshold))
    N=FP+TN
    P=TP+FN
    TPR=float(TP)/P
    FPR=float(FP)/N
    return TPR,FPR,pred_y_probs1


def draw_roc(x_train, x_test, y_train, y_test):
    """
    
        利用同一训练集和测试集，通过改变阈值，获得不同的tpr和fpr值，绘制roc曲线
        :param x_train:训练数据集自变量部分
        :param x_test:测试数据集自变量部分
        :param y_train:训练数据集因变量部分
        :param y_est:测试数据集因变量部分
        
    """
    #生成概率阈值
    thresholds = numpy.asarray([(10-j)/10.0 for j in range(10)])
    tprs=[];
    fprs = numpy.asarray([0. for j in range(10)])
    tprs = numpy.asarray([0. for j in range(10)])
    for j,thres in enumerate(thresholds):
        print "thres:",thres
        #利用逻辑斯蒂回归计算tpr,fpr
        tpr,fpr,pred_y_probs1=proc_by_logis(x_train, x_test, y_train, y_test,thres)
        fprs[j] = fpr
        tprs[j] = tpr
        print j,"tpr:",tpr
        print j,"fpr:",fpr
    #绘制ROC曲线
    plt.plot(fprs, tprs)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.show()


def calc_auc(x_train, x_test, y_train, y_test):
    """
    
        计算AUC值
        :param x_train:训练数据集自变量部分
        :param x_test:测试数据集自变量部分
        :param y_train:训练数据集因变量部分
        :param y_est:测试数据集因变量部分
        
    """
    from sklearn.metrics import roc_auc_score
    tpr,fpr,pred_y_probs1=proc_by_logis(x_train, x_test, y_train, y_test,0.5)
    obs=y_test['result'].values
    print type(obs),obs
    probs=pred_y_probs1['result'].values
    print type(probs),probs
    testing_auc = roc_auc_score(obs, probs)
    print("Example AUC: {auc}".format(auc=testing_auc))


if __name__ == '__main__':
    ##offline = processOfflineTrain('ccf_offline_stage1_train.csv')
    ### processOfflineUserID(offline)
    #by:zcs
    filename="./offlineTrainfeatures.csv"
    trainData = pd.read_csv(filename, header=0)
    x=trainData[["Distance","week","userRate","merchantRate","discountRate"]]
    x[["Distance","week"]]=x[["Distance","week"]].astype(int)
    x[["userRate","merchantRate","discountRate"]]=x[["userRate","merchantRate","discountRate"]].astype(float)
    y=trainData[["result"]].astype(int)

    #分割训练数据集
    x_train, x_test, y_train, y_test=choose_testdata(x,y,5,42)

    #绘制ROC曲线
    draw_roc(x_train, x_test, y_train, y_test)

    #计算AUC值
    calc_auc(x_train, x_test, y_train, y_test)

    print "##"*25

    #对测试集进行预测

