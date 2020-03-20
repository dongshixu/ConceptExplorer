import datetime
import time
import collections
import numpy as np
import sys
import math
import os
from random import randint
import threading
import sys
from drift_detector_xu.__init__ import *
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
import pandas as pd
from sklearn.naive_bayes import MultinomialNB


def timestamp_datetime(value):
    format = '%Y-%m-%d %H:%M:%S'
    # value为传入的值为时间戳(整形)，如：1332888820
    value = time.localtime(value)
    # # 经过localtime转换后变成
    # # time.struct_time(tm_year=2012, tm_mon=3, tm_mday=28, tm_hour=6, tm_min=53, tm_sec=40, tm_wday=2, tm_yday=88, tm_isdst=0)
    # 最后再经过strftime函数转换为正常日期格式。
    # dt = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(value))
    dt = time.strftime(format, value)
    return dt


def get_mean_var_std(arr):
    # 求均值
    arr_mean = np.mean(arr)
    # 求方差
    arr_var = np.var(arr)
    # 求标准差
    arr_std = np.std(arr, ddof=1)
    print("平均值为：%f" % arr_mean)
    print("方差为：%f" % arr_var)
    print("标准差为:%f" % arr_std)

    return [arr_mean, arr_var, arr_std]

def Order(n):
    global money
    money = money + n
    money = money - n
    print(money)

class thread(threading.Thread):
    def __init__(self, threadname):
        threading.Thread.__init__(self, name='线程' + threadname)
        self.threadname = int(threadname)

    def run(self):
        for i in range(1000000):
            Order(self.threadname)

class Producer(threading.Thread):
    def __init__(self, con):
        super(Producer, self).__init__()
        self._stop_event = threading.Event()
        self.lock_con = con

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()

    def run(self):
        global L
        global flag
        # while True:
        val = randint(0, 100)
        aa = []
        # print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        # print('生产者', self.name, ' Append'+str(val), L)
        for j in range(100):
            j += 1
            aa.append(j)
            if j % 7 == 0 or j == 100:
                self.lock_con.acquire()
                L.append(val)
                te(val)
                print('生产者', self.name, ' Append' + str(val), L, j)
                if j == 100:
                    print(aa)
                if len(L) == 5:
                    L = []
                    print('End', self.name, L)
                    self.lock_con.notifyAll()
                else:
                    # if j == 100:
                    #     print(self.isAlive(), "hello")
                    # else:
                    #     self.lock_con.wait()
                    self.lock_con.wait()
            else:
                print(self.name, j, self.isAlive())


def get_date(days):
    aa = time.strptime('1800-01-01', "%Y-%m-%d")
    datetime.date(aa[0], aa[1], aa[2])
    return datetime.date(aa[0], aa[1], aa[2]) + datetime.timedelta(days=days)


def getday(date):
    aa = time.strptime('1800-01-01', "%Y-%m-%d")  # '1800-01-01'
    bb = time.strptime(date, "%Y-%m-%d")
    return (datetime.datetime(bb[0], bb[1], bb[2])-datetime.datetime(aa[0], aa[1], aa[2])).days
# class Consumer(threading.Thread):
#     def run(self):
#         global L
#         while True:
#             lock_con.acquire()
#             if len(L) == 0:
#                 lock_con.wait()
#             print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
#             print('消费者', self.name, 'Delete'+str(L[0]), L)
#             del L[0]
#             lock_con.release()
#             time.sleep(0.5)
#
# class Test(threading.Thread):
#     def run(self):
#         global L
#         # lock_con.acquire()
#         # if len(L) == 0:
#         #     lock_con.wait()
#         # print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
#         # print('消费者', self.name, 'Delete'+str(L[0]), L)
#         # del L[0]
#         # lock_con.release()
#         # time.sleep(0.5)
#         lock_con.acquire()
#         for k in range(10):
#             L.append(k)
#         lock_con.release()


if __name__ == '__main__':
    # date = datetime.date(2013, 1, 4)
    # T = datetime.time(11)
    # t = str(date) + ' ' + str(T)
    # # r = int(time.mktime((2013, 1, 4, 10)))
    # st_ts = time.mktime(time.strptime(t, '%Y-%m-%d %H:%M:%S'))
    # print(st_ts)
    # d1 = collections.OrderedDict()
    #
    # d1['a'] = 1
    #
    # d1['b'] = 2
    #
    # d1['c'] = 3
    #
    # d1['1'] = 4.0
    #
    # d1['2'] = 1
    # a = np.dot(np.array(list(d1.values())), 2/3)
    # b = np.dot(np.array(list(d1.values())), 2/3)
    # c = a + b
    # print(np.zeros(5) + c)
    # print(np.dot(np.array(list(d1.values())), 2/3))
    #
    # pr = True
    # pr = 1 if pr is False else 0
    # print(pr)

    # drift_detector = DDM()
    #
    # for i in range(100):
    #     if i % 2 == 0:
    #         prediction_status = True
    #     else:
    #         prediction_status = False
    #     warning_status, drift_status = drift_detector.detect(prediction_status)
    #     print(drift_detector.warning_level)
    # print(drift_detector.P, drift_detector.S)
    #
    # a = []
    # for i in range(100):
    #     if i % 2 == 0:
    #         a.append(True)
    #     else:
    #         a.append(False)
    #
    # for status in a:
    #     warning_status, drift_status = drift_detector.detect(status)
    #     print(drift_detector.warning_level, drift_detector.drift_level, drift_detector.current_level)
    #     print(drift_detector.P, drift_detector.S)
    # print(drift_detector.NUM_INSTANCES_SEEN)
    # print(drift_detector.P, drift_detector.S)

    # x = [0.122, 0.34, 0.988, 1.0, 0]
    # y = [1, 0, 1, 1, 0]
    # d1 = np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
    # print(d1)
    # a = [[1, 2, 3, 4],
    #      [4, 3, 2, 1]]
    # a = np.array(a)
    # print(a[:, 1].tolist())
    # print([[0]*10][0])
    #
    # print(timestamp_datetime(1363867200 + 4*3600))
    # print(1363867200 + 4*3600)
    #
    # path = 'E:/zju/result/prsa_data_20200214_202253'
    # for x, c, v in os.walk(path):
    #     print(x)
    #     print(c)
    #     print(v)
    # print(os.path.dirname(path))

    # for i in range(np.shape(a)[1]-1):
    #     print(i)
    #     print(a[:, i])
    #     d1 = np.dot(a[:, i], a[:, -1]) / (np.linalg.norm(a[:, i]) * np.linalg.norm(a[:, -1]))
    #     print(d1)
    #
    # aa = [1, 2, 3, 4]
    # print(aa[0])
    # print(aa[0:4])
    # print(timestamp_datetime(1393329600))
    # print((1393329600-1363867200) / (24 * 3600))
    # print(aa[:-1])
    #
    # bb = [1, 4, 5, 3, 3, 4, 1, 2, 2, 1]
    # print(sorted(set(bb)))
    #
    # print([i*0.1 for i in range(0, 11)])
    # cc = {}
    # print(len(cc.keys()))
    #
    # dd = np.array([1, 2, 3, 2, 1])
    #
    # print(np.min(abs(dd-2)))
    # print()

    # money = 0
    # t1 = thread('1')
    # t2 = thread('5')
    # t1.start()
    # t2.start()
    # t1.join()
    # t2.join()
    # print(money)

    # L = []
    # s = 0
    #
    # def te(jj):
    #     global s
    #     s += jj
    #     print("Test function result is:", s)
    #
    # def run():
    #     flag = False
    #     lock_con = threading.Condition()
    #     threads = []
    #     for i in range(5):
    #         threads.append(Producer(lock_con))
    #     for t in threads:
    #         t.daemon = 1
    #         t.start()
    #     for t in threads:
    #         t.join()
    #     while True:
    #         if flag:
    #             break
    #     print("Hello world")
    #
    # # run()
    # # print("Hello world")
    #
    # num = [1, 2, 10, 42, 42, 42, 43, 43, 44]
    # ct = 0
    # start = 0
    # interval = 7
    # a = 0
    # zz = []
    # start_list = [0]
    # for value in num:
    #     d_v = value - start
    #     if d_v >= interval:
    #         # 计算每个batch的值
    #         inter = int(d_v / interval)
    #         for i in range(inter):
    #             start += interval
    #             start_list.append(start)
    #             zz.append(a)
    #             a = 0
    #     a += value
    #
    #     if value == num[-1]:
    #         zz.append(a)
    #
    # print(zz, start_list)
    #
    # print(int(32/7))
    #
    # path = 'E:/zju/result'
    # name = None
    #
    # print((1362585600-1361980800)/(7*24*3600))
    print(datetime.datetime.fromtimestamp(1365912000))
    #
    # ddd = [1, 2, 3, 4]
    # print(ddd[2:])
    #
    # print(get_date(78616))
    # print(getday('2015-03-31'))
    # # for i in range(2):
    # #     ddd.pop(0)
    # #     print(ddd)
    # print(ddd[2:])
    #
    # vvv = {"1234": 'haha', "bb": 'hihi'}
    # # for i in range(10):
    # #     if i == 3:
    # #         print("Hello vag")
    # #         break
    # #     print(i)
    # if 'aa' in vvv:
    #     print("hello")
    # if '2013-8-16' <= '2014-03-31' <= '2014-1-19':
    #     print("nihao ")
    #
    # print(int(time.mktime(time.strptime('2013-8-16' + ' ' + '00:00:00', '%Y-%m-%d %H:%M:%S'))))
    # print(int(time.mktime(time.strptime('2014-1-19' + ' ' + '00:00:00', '%Y-%m-%d %H:%M:%S'))))
    #
    # vvv[12] = 12
    # if 12 in vvv:
    #     vvv[12] += 10
    # print('12' in vvv)
    # aaa = ['12', '23']
    # print(list(map(int, aaa)))
    # nums = [1, 1, 2, 6, 9, 10]
    # newnums = len(list(filter(lambda x: 2 <= x <= 5, nums)))
    # print(newnums)

    # from pgmpy.estimators import HillClimbSearch, BicScore
    # # values = pd.DataFrame(np.random.randint(low=0, high=3, size=(11, 4)),  columns=['A', 'B', 'C', 'D'])
    # # values.drop('E', axis=1, inplace=True)
    # aaa = [[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 0], [1, 1, 0, 0]]
    # bbb = [[0, 0, 0, 1, 1, 1, 0, 1], [1, 2, 3, 1, 2, 3, 3, 1], [1, 2, 3, 1, 2, 3, 2, 3], [1, 2, 3, 1, 2, 3, 3, 2]]
    # # bbb = [[0, 0, 0, 1, 1, 1, 0], [0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 1, 1, 1, 0], [1, 2, 3, 1, 2, 3, 2], [1, 2, 3, 1, 2, 3, 3], [1, 2, 3, 1, 2, 3, 2], [1, 2, 3, 1, 2, 3, 3]]
    # bbb = np.array(bbb).T
    # # values = pd.DataFrame(bbb,  columns=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'])
    # values = pd.DataFrame(bbb, columns=['A', 'B', 'C', 'D'])
    # # test_data = pd.DataFrame([1, 2, 3],  columns=['A', 'B', 'C', 'D'])
    # # print(values)
    #
    # train_data = values[:-1].copy()
    # est = HillClimbSearch(values, scoring_method=BicScore(train_data))
    # best_model = est.estimate()
    # edges = best_model.edges()

    # train_data1 = values[:2]
    # train_data2 = values[2:4]
    # predict_data = values[-1:]
    # predict_data1 = predict_data.copy()
    # predict_data2 = predict_data.copy()
    # predict_data3 = predict_data.copy()

    # [('B', 'A'), ('C', 'A'), ('D', 'A')]
    #
    # model = BayesianModel(edges)
    # model.add_nodes_from(['A', 'B', 'C', 'D'])
    #
    # model.fit(train_data)

    # predict_data1.drop('A', axis=1, inplace=True)
    # print(model.predict_probability(predict_data1))
    # print(model.get_independencies())
    # test_data = values[-1:].copy()
    # for cpd in model.get_cpds():
    #     print("CPD of {variable}:".format(variable=cpd.variable))
    #     print(cpd)
    # test_data.drop('A', axis=1, inplace=True)
    # test_data.drop('B', axis=1, inplace=True)
    # # print(model.get_independencies())
    # print(model.get_cpds("B"))
    # print(model.predict_probability(test_data[-1:]))

    # student = BayesianModel([('diff', 'grade'), ('intel', 'grade'), ('haha', 'grade')])
    # cpd = TabularCPD('grade', 2, [[0.1, 0.9, 0.2, 0.7],
    #                               [0.9, 0.1, 0.8, 0.3]],
    #                  ['intel', 'diff'], [2, 2])
    # cpd1 = TabularCPD('grade', 2, [[0.1, 0.9, 0.2, 0.7],
    #                               [0.9, 0.1, 0.8, 0.3]],
    #                  ['intel', 'haha'], [2, 2])
    #
    # student.add_cpds(cpd)
    # student.add_cpds(cpd1)
    # student.get_cpds()
    # for cpd in student.get_cpds():
    #     print("CPD of {variable}:".format(variable=cpd.variable))
    #     print(cpd)

    # model.fit(train_data2)
    # predict_data2.drop('A', axis=1, inplace=True)
    # print(model.predict_probability(predict_data2))
    # # print(model.get_cpds("A"))
    #
    # model.fit(values[:4])
    # predict_data3.drop('A', axis=1, inplace=True)
    # print(model.predict_probability(predict_data3))
    # print(model.get_cpds("A"))
    # predict_data = predict_data.copy()
    # predict_data.drop('A', axis=1, inplace=True)
    # # y_pred = model.predict(predict_data)
    # y_pred = model.predict_probability(predict_data)
    # print(y_pred)

    # aaa = np.array(aaa)
    # print(aaa)
    # print(aaa[:-1, :-1])
    # print(aaa[:-1, -1:].T)
    # unix_time = int(time.mktime(time.strptime('2014-01-19' + ' ' + '00:00:00', '%Y-%m-%d %H:%M:%S')))
    # print(unix_time)
    # interval = 1
    # start_time = 1
    # temp_num = []
    # count = 0
    # for i in [1, 2, 3, 4, 4, 5, 5, 5, 10, 11]:
    #     if i > start_time:
    #         diff = int((i - start_time) / interval)
    #         for _ in range(diff):
    #             start_time += interval
    #             temp_num.append(count)
    #             count = 0
    #         count += 1
    #     else:
    #         count += 1
    #     if i == 11:
    #         temp_num.append(count)
    # print(temp_num)
    # print(sum(temp_num), len([1, 2, 3, 4, 4, 5, 5, 5, 10, 11]))

    # arr = [[0, 1, 2], [0, 1, 2], [2, 0, 2]]
    # lab = [0, 1, 1]
    # arr_t = [[1, 2, 2]]
    # lab_t = [1]
    # clf = MultinomialNB()
    # clf.fit(arr, lab)
    # print(clf.predict(arr_t), clf.predict_proba(arr_t))
    print(1365912000 + 24 * 3600)
    print(1366401600 + 24 * 3600)






