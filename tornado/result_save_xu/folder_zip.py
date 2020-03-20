"""
@Author: Dustin Xu
@Date: 2020/2/14 20:26 PM
@Description: Data folder zip
"""
import os
import zipfile
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import json
import numpy as np
import time
import datetime

class Zip:
    def __init__(self, file_path):
        self.zipDir(file_path)

    @staticmethod
    def zipDir(aim_folder):
        _out_dir = os.path.dirname(aim_folder) + '/' + aim_folder.split('/')[-1] + '.zip'
        _zip = zipfile.ZipFile(_out_dir, "w", zipfile.ZIP_DEFLATED)
        for path, dir_names, file_names in os.walk(aim_folder):
            # 去掉目标跟路径，只对目标文件夹下边的文件及文件夹进行压缩
            f_path = path.replace(aim_folder, '')

            for filename in file_names:
                _zip.write(os.path.join(path, filename), os.path.join(f_path, filename))
        _zip.close()
        print('{} saved !'.format(_out_dir))

    @staticmethod
    def plot(data):
        color = ['yellow', 'red', 'blue', 'green']
        for j, name in enumerate(data.keys()):
            x = [i for i in range(len(data[name]))]
            plt.plot(x, data[name], c=color[j], label=name)
        plt.legend(list(data.keys()))
        plt.show()

    @staticmethod
    def plot_multi(data):
        color = ['yellow', 'red', 'blue', 'green']
        for j, name in enumerate(data.keys()):
            x = [i for i in range(len(data[name]['accuracy']))]
            plt.plot(x, data[name]['accuracy'], c=color[j], label=name)
        plt.legend(list(data.keys()))
        plt.show()

    @staticmethod
    def plot_multi_1(data, name):
        color = ['yellow', 'red', 'blue', 'green']
        div = len(data['prob']) - (len(data['prob']) % 100) + 100
        for j in range(0, div, 100):
            if j+100 >= len(data['prob']):
                end = len(data['prob'])
            else:
                end = j + 100
            x = [i for i in range(j, end)]
            fig, ax1 = plt.subplots()
            ax2 = ax1.twinx()  # 做镜像处理
            ax1.plot(x, data['prob'][j:end], 'r-')
            ax2.plot(x, data['ground_truth'][j:end], '#1F77B4')

            # x_major_locator = MultipleLocator(10)

            ax1.set_xlabel('Batch Num*2')  # 设置x轴标题
            ax1.set_ylabel('drift prob', color='r')  # 设置Y1轴标题
            ax2.set_ylabel('ground truth', color='#1F77B4')  # 设置Y2轴标题
            # ax1.xaxis.set_major_locator(x_major_locator)
            # ax2.xaxis.set_major_locator(x_major_locator)
            plt.title(name)

            # for j, name in enumerate(data.keys()):
            #     x = [i for i in range(len(data[name]['accuracy']))]
            #     plt.plot(x, data[name]['accuracy'], c=color[j], label=name)
            # plt.legend(list(data.keys()))
            # plt.legend(['drift prob', 'ground truth'])
            plt.show()

    @staticmethod
    def plot_multi_2(data, name):
        color = ['yellow', 'red', 'blue', 'green']
        x = [i for i in range(len(data))]
        fig, ax1 = plt.subplots()
        ax1.plot(x, data)
        ax1.plot(x, data['y3'])
        ax1.plot(x, data['y4'], c='r')

        ax1.set_xlabel('bate')  # 设置x轴标题
        ax1.set_ylabel('num', color='r')  # 设置Y1轴标题
        plt.title(name)

        plt.show()

    @staticmethod
    def plot_multi_3(data, name):
        color = ['yellow', 'red', 'blue', 'green']
        x = [i for i in range(len(data['y3']))]
        fig, ax1 = plt.subplots()
        ax1.plot(x, data['y3'])
        ax1.plot(x, data['y4'], c='r')

        ax1.set_xlabel('Batch')  # 设置x轴标题
        ax1.set_ylabel('Accuracy', color='r')  # 设置Y1轴标题
        plt.title(name)
        plt.legend(['NB', 'bayes network'])
        plt.show()


if __name__ == '__main__':
    path_data = 'E:/zju/movie_data/movie_data/movie_lens/data_0.npy'
    path_label = 'E:/zju/movie_data/movie_data/movie_lens/label_0.npy'
    file_path = 'E:/zju/result/movie_data_result_20200304_120407/data_num.json'
    path_nb4 = 'E:/zju/result/prsa_data_result_20200305_195546experiment_with_the_figure.json'  # bn
    path_nb3 = 'E:/zju/result/prsa_data_result_20200305_232640experiment_with_the_figure.json'  # nb
    path_nb31 = 'E:/zju/result/prsa_data_result_20200301_142841experiment_with_the_figure.json'
    path_nb_m = 'E:/zju/result/movie_data_result_20200305_234835experiment_with_the_figure.json'
    path_nb_p = 'E:/zju/result/prsa_data_result_20200307_091413experiment_with_the_figure.json'
    path_nb_n = 'E:/zju/result/netease_data_result_20200306_134836experiment_with_the_figure.json'
    path_nb_new = 'E:/zju/result/prsa_data_result_20200306_212144experiment_with_the_figure.json'
    # path_nb4 = 'E:/zju/result/prsa_data_result_20200301_142841experiment_with_the_figure.json'
    # path_data = 'C:/Users/徐懂事/Desktop/zj/air/Wanshouxigong/data_0.npy'
    # path_label = 'C:/Users/徐懂事/Desktop/zj/air/Wanshouxigong/label_0.npy'

    # file3 = 0
    # file4 = 0
    # accuracy = {}
    #
    # with open(path_nb3, 'r', encoding='UTF-8') as f:
    #     file3 = json.load(f)
    # with open(path_nb4, 'r', encoding='UTF-8') as f:
    #     file4 = json.load(f)
    #
    # print(file3)
    # print(file4)
    # for key in file3.keys():
    #     right_count3 = 0
    #     all_count3 = 0
    #     right_count4 = 0
    #     all_count4 = 0
    #     accuracy[key] = dict(y3=[], y4=[])
    #     for i, value in enumerate(file3[key]['prob']):
    #
    #         if file3[key]['ground_truth'][i] >= 3:
    #             gt3 = 1
    #         else:
    #             gt3 = 0
    #         if file3[key]['prob'][i] >= 0.5:
    #             pre3 = 1
    #         else:
    #             pre3 = 0
    #
    #         # print(file3[key]['ground_truth'][i], file3[key]['prob'][i], gt3, pre3)
    #
    #         if file4[key]['ground_truth'][i] >= 3:
    #             gt4 = 1
    #         else:
    #             gt4 = 0
    #         if file4[key]['prob'][i] >= 0.5:
    #             pre4 = 1
    #         else:
    #             pre4 = 0
    #         # print(file4[key]['ground_truth'][i], file4[key]['prob'][i], gt4, pre3)
    #
    #         if gt3 == pre3:
    #             right_count3 += 1
    #         else:
    #             print('3wrong')
    #         if gt4 == pre4:
    #             right_count4 += 1
    #         else:
    #             print("4wrong")
    #         all_count3 += 1
    #         all_count4 += 1
    #         accuracy[key]['y3'].append(round(right_count3 / all_count3, 4))
    #         accuracy[key]['y4'].append(round(right_count4 / all_count4, 4))
    #
    # for key in accuracy.keys():
    #     Zip.plot_multi_3(accuracy[key], key)

    with open(path_nb_p, 'r', encoding='UTF-8') as f:
        result = json.load(f)
    #     print(result.keys())
    #     for file_name in result.keys():
    #         Zip.plot_multi_1(result[file_name], file_name)
        for file_name in result.keys():
            Zip.plot_multi_1(result[file_name], file_name)
            print(len(result[file_name]['ground_truth']))
    #         print(file_name, sum(result[file_name][:5]), sum(result[file_name]))
    #         print(result[file_name][:5])
    #         print('Average data volume per batch:{}'.format(file_name), int(sum(result[file_name])/len(result[file_name])))
    #         print('Median data volume per batch:{}'.format(file_name), np.median(np.array(result[file_name])))
    #         Zip.plot_multi_2(result[file_name], file_name)

    data = np.load(path_data)
    # date = list(data[0][:4])
    # print(date)
    # date_time = list(map(int, date[:4]))
    # d = datetime.date(date_time[0], date_time[1], date_time[2])
    # tt = datetime.time(date_time[3])
    # datetime_str = str(d) + ' ' + str(tt)
    # unix_time = int(time.mktime(time.strptime(datetime_str, '%Y-%m-%d %H:%M:%S')))
    # print(unix_time)
    label = np.load(path_label)
    # print(len(label))
    # print(type(data))
    # print(len(data))
    date_count = [0, 0, 0, 0, 0]
    for re in data:
        if re[-1] == 77855:
            date_count[0] += 1
        elif re[-1] == 77856:
            date_count[1] += 1
        elif re[-1] == 77857:
            date_count[2] += 1
        elif re[-1] == 77858:
            date_count[3] += 1
        elif re[-1] == 77859:
            date_count[4] += 1
    print(date_count)
    print(data)
    print(len(label))
    print(len(data))
    # print([value[-1] for value in data[:100]])
    # new_data = data[400:]
    # new_label = label[400:]
    # new_data = data[2000:]
    # new_label = label[2000:]
    # new_data = data[6000:]
    # new_label = label[6000:]
    # print(len(new_data), len(new_label))
    # print(type(new_data), type(new_label))
    # np.save('E:/zju/result/movie_data_result_20200304_091727/movie_lens/label_0.npy', new_label)
    # np.save('E:/zju/result/movie_data_result_20200304_091727/movie_lens/data_0.npy', new_data)

    # print(data[-3:])
