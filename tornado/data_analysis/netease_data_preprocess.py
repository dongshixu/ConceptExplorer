"""
@Author: Dustin Xu
@Date: 2020/2/29 12:23 PM
@Description:  preprocessor netease data

"""
import json
import os
import time
import numpy as np

"""
three servers
17, 164, 230
"""

class NetEase:
    def __init__(self):
        self.servers = dict(s17=dict(id=[], data=[], friendship=[], statistic=dict(maxNum=0, avg=0, count=[])), s164=dict(id=[], data=[], friendship=[], statistic=dict(maxNum=0, avg=0, count=[])), s230=dict(id=[], data=[], friendship=[], statistic=dict(maxNum=0, avg=0, count=[])))
        self.data_dict = dict(s17=[], s164=[], s230=[])
        self.temp_dict = {}
        self.time_interval = 3600 * 24

    @staticmethod
    def _find_sub_file(path):
        return os.listdir(path)

    @staticmethod
    def _save(sp, data):
        with open(sp, 'w') as file_obj:
            json.dump(data, file_obj)
            print(sp + '==>' + 'saved ok!')

    def get_player(self, attr_path, attr_save_path):
        for filename in self._find_sub_file(attr_path):
            file_name = attr_path + '/' + filename
            with open(file_name, 'rb') as f:
                for result in f:
                    result = result.decode('utf-8').strip('/').split()
                    # print(result)
                    if int(result[1]) == 17:
                        if result[0] not in self.servers['s17']['id']:
                            self.servers['s17']['id'].append(result[0])
                    elif int(result[1]) == 164:
                        if result[0] not in self.servers['s164']['id']:
                            self.servers['s164']['id'].append(result[0])
                    elif int(result[1]) == 230:
                        if result[0] not in self.servers['s230']['id']:
                            self.servers['s230']['id'].append(result[0])

            print(file_name, "done!")
        self._save(attr_save_path, self.servers)  # # dealt after

    # def get_friends(self, json_file, friend_path, friend_save_path):
    #     with open(json_file, 'r', encoding='utf-8') as file:
    #         result_json = json.load(file)
    #     print(len(result_json['s17']['id']), len(result_json['s164']['id']), len(result_json['s230']['id']))
    #     for filename in self._find_sub_file(friend_path):
    #         file_name = friend_path + '/' + filename
    #         count = 0
    #         with open(file_name, 'rb') as f:
    #             for result in f:
    #                 count += 1
    #                 if count % 100000 == 0:
    #                     print(count)
    #                 result = result.decode('utf-8').strip('/').split()
    #                 for key in result_json.keys():
    #                     if result[0] in result_json[key]['id']:
    #                         fd = result[-1].split(',')
    #                         length = len(fd)
    #                         self.temp_dict[result[0]] = fd
    #                         result_json[key]['friendship'].append(self.temp_dict)
    #                         result_json[key]['length'] = length
    #                         result_json[key]['statistic']['count'].append(length)
    #                         self.temp_dict = {}
    #                         break
    #     self._save(friend_save_path, result_json)  # # dealt after

    def get_friends(self, json_file, friend_path, friend_save_path):
        with open(json_file, 'r', encoding='utf-8') as file:
            result_json = json.load(file)
        print(len(result_json['s17']['id']), len(result_json['s164']['id']), len(result_json['s230']['id']))
        for filename in self._find_sub_file(friend_path):
            file_name = friend_path + '/' + filename
            count = 0
            with open(file_name, 'rb') as f:
                for result in f:
                    count += 1
                    if count % 100000 == 0:
                        print(count)
                    result = result.decode('utf-8').strip('/').split()
                    if int(result[1]) == 17:
                        fd = result[-1].split(',')
                        self.temp_dict[result[0]] = fd
                        self.temp_dict['length'] = len(fd)
                        result_json['s17']['friendship'].append(self.temp_dict)
                    elif int(result[1]) == 164:
                        fd = result[-1].split(',')
                        self.temp_dict[result[0]] = fd
                        self.temp_dict['length'] = len(fd)
                        result_json['s164']['friendship'].append(self.temp_dict)
                    elif int(result[1]) == 230:
                        fd = result[-1].split(',')
                        self.temp_dict[result[0]] = fd
                        self.temp_dict['length'] = len(fd)
                        result_json['s230']['friendship'].append(self.temp_dict)
                    self.temp_dict = {}

        self._save(friend_save_path, result_json)  # # dealt after

    def get_consumption(self, json_file, consumption_path, consumption_save_path):
        with open(json_file, 'r', encoding='utf-8') as file:
            result_json = json.load(file)
        for key in result_json.keys():
            result_json[key]['consumption'] = {}
        print(len(result_json['s17']['id']), len(result_json['s164']['id']), len(result_json['s230']['id']))
        for filename in self._find_sub_file(consumption_path):
            file_name = consumption_path + '/' + filename
            count = 0
            with open(file_name, 'rb') as f:
                for result in f:
                    count += 1
                    if count % 100000 == 0:
                        print(count)
                    result = result.decode('utf-8').strip('/').split()
                    unix_time = int(time.mktime(time.strptime(result[0] + ' ' + '00:00:00', '%Y-%m-%d %H:%M:%S')))
                    if 1376582400 <= unix_time <= 1390060800:
                        unix_time = str(unix_time)
                        if int(result[2]) == 17:
                            if result[3] in result_json['s17']['id']:
                                if result[3] in result_json['s17']['consumption']:
                                    if unix_time in result_json['s17']['consumption'][result[3]]:
                                        result_json['s17']['consumption'][result[3]][unix_time].append(float(result[-1]))
                                    else:
                                        result_json['s17']['consumption'][result[3]][unix_time] = [float(result[-1])]
                                else:
                                    result_json['s17']['consumption'][result[3]] = dict()
                                    result_json['s17']['consumption'][result[3]][unix_time] = [float(result[-1])]
                        elif int(result[2]) == 164:
                            if result[3] in result_json['s164']['id']:
                                if result[3] in result_json['s164']['consumption']:
                                    if unix_time in result_json['s164']['consumption'][result[3]]:
                                        result_json['s164']['consumption'][result[3]][unix_time].append(float(result[-1]))
                                    else:
                                        result_json['s164']['consumption'][result[3]][unix_time] = [float(result[-1])]
                                else:
                                    result_json['s164']['consumption'][result[3]] = dict()
                                    result_json['s164']['consumption'][result[3]][unix_time] = [float(result[-1])]
                        elif int(result[2]) == 230:
                            if result[3] in result_json['s230']['consumption']:
                                if unix_time in result_json['s230']['consumption'][result[3]]:
                                    result_json['s230']['consumption'][result[3]][unix_time].append(float(result[-1]))
                                else:
                                    result_json['s230']['consumption'][result[3]][unix_time] = [float(result[-1])]
                            else:
                                result_json['s230']['consumption'][result[3]] = dict()
                                result_json['s230']['consumption'][result[3]][unix_time] = [float(result[-1])]

        self._save(consumption_save_path, result_json)  # # dealt after

    def get_log(self, json_file, log_path, log_save_path):
        with open(json_file, 'r', encoding='utf-8') as file:
            result_json = json.load(file)
        for key in result_json.keys():
            result_json[key]['log'] = {}
        for filename in self._find_sub_file(log_path):
            file_name = log_path + '/' + filename
            count = 0
            with open(file_name, 'rb') as f:
                for result in f:
                    count += 1
                    if count % 100000 == 0:
                        print(count)
                    result = result.decode('utf-8').strip('/').split()
                    if result[1] in ['17', '164', '230']:
                        unix_time = int(time.mktime(time.strptime(result[2] + ' ' + '00:00:00', '%Y-%m-%d %H:%M:%S')))
                        if 1376582400 <= unix_time <= 1390060800:
                            if int(result[1]) == 17:
                                if result[0] in result_json['s17']['id']:
                                    if result[0] in result_json['s17']['log']:
                                        if unix_time in result_json['s17']['log'][result[0]]:
                                            result_json['s17']['log'][result[0]][unix_time].append(int(result[-1]))
                                        else:
                                            result_json['s17']['log'][result[0]][unix_time] = [int(result[-1])]
                                    else:
                                        result_json['s17']['log'][result[0]] = dict()
                                        result_json['s17']['log'][result[0]][unix_time] = [int(result[-1])]
                            elif int(result[1]) == 164:
                                if result[0] in result_json['s164']['id']:
                                    if result[0] in result_json['s164']['log']:
                                        if unix_time in result_json['s164']['log'][result[0]]:
                                            result_json['s164']['log'][result[0]][unix_time].append(int(result[-1]))
                                        else:
                                            result_json['s164']['log'][result[0]][unix_time] = [int(result[-1])]
                                    else:
                                        result_json['s164']['log'][result[0]] = dict()
                                        result_json['s164']['log'][result[0]][unix_time] = [int(result[-1])]
                            elif int(result[1]) == 230:
                                if result[0] in result_json['s230']['id']:
                                    if result[0] in result_json['s230']['log']:
                                        if unix_time in result_json['s230']['log'][result[0]]:
                                            result_json['s230']['log'][result[0]][unix_time].append(int(result[-1]))
                                        else:
                                            result_json['s230']['log'][result[0]][unix_time] = [int(result[-1])]
                                    else:
                                        result_json['s230']['log'][result[0]] = dict()
                                        result_json['s230']['log'][result[0]][unix_time] = [int(result[-1])]
        self._save(log_save_path, result_json)

    def get_pvp(self, json_file, pvp_path, pvp_save_path):
        with open(json_file, 'r', encoding='utf-8') as file:
            result_json = json.load(file)
        for key in result_json.keys():
            result_json[key]['pvp'] = {}
        for filename in self._find_sub_file(pvp_path):
            file_name = pvp_path + '/' + filename
            count = 0
            with open(file_name, 'rb') as f:
                for result in f:
                    count += 1
                    if count % 100000 == 0:
                        print(count)
                    result = result.decode('utf-8').strip('/').split()
                    if result[2] in ['17', '164', '230']:
                        unix_time = int(time.mktime(time.strptime(result[0] + ' ' + '00:00:00', '%Y-%m-%d %H:%M:%S')))
                        if 1376582400 <= unix_time <= 1390060800:
                            if int(result[2]) == 17:
                                if result[3] in result_json['s17']['id']:
                                    if result[3] in result_json['s17']['pvp']:
                                        result_json['s17']['pvp'][result[3]].append(unix_time)
                                    else:
                                        result_json['s17']['pvp'][result[3]] = [unix_time]
                            elif int(result[2]) == 164:
                                if result[3] in result_json['s164']['id']:
                                    if result[3] in result_json['s164']['pvp']:
                                        result_json['s164']['pvp'][result[3]].append(unix_time)
                                    else:
                                        result_json['s164']['pvp'][result[3]] = [unix_time]
                            elif int(result[2]) == 230:
                                if result[3] in result_json['s230']['id']:
                                    if result[3] in result_json['s230']['pvp']:
                                        result_json['s230']['pvp'][result[3]].append(unix_time)
                                    else:
                                        result_json['s230']['pvp'][result[3]] = [unix_time]
        self._save(pvp_save_path, result_json)

    def get_paid(self, json_file, paid_path, paid_save_path):
        with open(json_file, 'r', encoding='utf-8') as file:
            result_json = json.load(file)
        for key in result_json.keys():
            result_json[key]['paid'] = {}
        for filename in self._find_sub_file(paid_path):
            file_name = paid_path + '/' + filename
            count = 0
            with open(file_name, 'rb') as f:
                for result in f:
                    count += 1
                    if count % 100000 == 0:
                        print(count)
                    result = result.decode('utf-8').strip('/').split()
                    unix_time = int(time.mktime(time.strptime(result[0] + ' ' + '00:00:00', '%Y-%m-%d %H:%M:%S')))
                    if 1376582400 <= unix_time <= 1390060800:
                        if int(result[2]) == 17:
                            if result[3] in result_json['s17']['id']:
                                if result[3] in result_json['s17']['paid']:
                                    result_json['s17']['paid'][result[3]].append(unix_time)
                                else:
                                    result_json['s17']['paid'][result[3]] = [unix_time]
                        elif int(result[2]) == 164:
                            if result[3] in result_json['s164']['id']:
                                if result[3] in result_json['s164']['paid']:
                                    result_json['s164']['paid'][result[3]].append(unix_time)
                                else:
                                    result_json['s164']['paid'][result[3]] = [unix_time]
                        elif int(result[2]) == 230:
                            if result[3] in result_json['s230']['id']:
                                if result[3] in result_json['s230']['paid']:
                                    result_json['s230']['paid'][result[3]].append(unix_time)
                                else:
                                    result_json['s230']['paid'][result[3]] = [unix_time]
        self._save(paid_save_path, result_json)

    def get_chat(self, json_file, chat_path, chat_save_path):
        with open(json_file, 'r', encoding='utf-8') as file:
            result_json = json.load(file)
        for key in result_json.keys():
            result_json[key]['chat'] = {}
        for sub in ['17', '164', '230']:
            chat_path_u = chat_path + '/{}'.format(sub)
            for filename in self._find_sub_file(chat_path_u):
                file_name = chat_path_u + '/' + filename
                count = 0
                with open(file_name, 'rb') as f:
                    for result in f:
                        count += 1
                        if count % 100000 == 0:
                            print(count)
                        result = result.decode('utf-8').strip('/').split()
                        result = result[:1] + result[-1].split(',')
                        unix_time = int(time.mktime(time.strptime(result[0] + ' ' + '00:00:00', '%Y-%m-%d %H:%M:%S')))
                        if 1376582400 <= unix_time <= 1390060800:
                            if int(sub) == 17:
                                if result[2] in result_json['s17']['chat']:
                                    result_json['s17']['chat'][result[2]].append(unix_time)
                                else:
                                    result_json['s17']['chat'][result[2]] = [unix_time]
                            elif int(sub) == 164:
                                if result[2] in result_json['s164']['chat']:
                                    result_json['s164']['chat'][result[2]].append(unix_time)
                                else:
                                    result_json['s164']['chat'][result[2]] = [unix_time]
                            elif int(sub) == 230:
                                if result[2] in result_json['s230']['chat']:
                                    result_json['s230']['chat'][result[2]].append(unix_time)
                                else:
                                    result_json['s230']['chat'][result[2]] = [unix_time]
                print(filename, "end!")
        self._save(chat_save_path, result_json)

    def get_label(self, json_file, label_path, label_save_path):
        with open(json_file, 'r', encoding='utf-8') as file:
            result_json = json.load(file)
        for key in result_json.keys():
            result_json[key]['label'] = {}
        for filename in self._find_sub_file(label_path):
            file_name = label_path + '/' + filename
            count = 0
            with open(file_name, 'rb') as f:
                for result in f:
                    count += 1
                    if count % 100000 == 0:
                        print(count)
                    result = result.decode('utf-8').strip('/').split()
                    # print(result)
                    if result[2] in ['17', '164', '230']:
                        unix_time = int(time.mktime(time.strptime(result[0] + ' ' + '00:00:00', '%Y-%m-%d %H:%M:%S')))
                        if 1376582400 <= unix_time <= 1390060800:
                            if int(result[-2]) == 1:
                                if int(result[2]) == 17:
                                    if result[3] in result_json['s17']['label']:
                                        if unix_time in result_json['s17']['label'][result[3]]:
                                            result_json['s17']['label'][result[3]][unix_time].append(result[-1])
                                        else:
                                            result_json['s17']['label'][result[3]][unix_time] = [result[-1]]
                                    else:
                                        result_json['s17']['label'][result[3]] = dict()
                                        result_json['s17']['label'][result[3]][unix_time] = [result[-1]]
                                elif int(result[2]) == 164:
                                    if result[3] in result_json['s164']['label']:
                                        if unix_time in result_json['s164']['label'][result[3]]:
                                            result_json['s164']['label'][result[3]][unix_time].append(result[-1])
                                        else:
                                            result_json['s164']['label'][result[3]][unix_time] = [result[-1]]
                                    else:
                                        result_json['s164']['label'][result[3]] = dict()
                                        result_json['s164']['label'][result[3]][unix_time] = [result[-1]]
                                elif int(result[2]) == 230:
                                    if result[3] in result_json['s230']['label']:
                                        if unix_time in result_json['s230']['label'][result[3]]:
                                            result_json['s230']['label'][result[3]][unix_time].append(result[-1])
                                        else:
                                            result_json['s230']['label'][result[3]][unix_time] = [result[-1]]
                                    else:
                                        result_json['s230']['label'][result[3]] = dict()
                                        result_json['s230']['label'][result[3]][unix_time] = [result[-1]]
        self._save(label_save_path, result_json)

    def combine_features(self, total_path):
        # attr_path = '/home/fhz_11821062/czx_21821062/xds/netease/player_daily_attr'
        attr_path = 'E:/zju/netease/player_daily_attr'
        # friends_path = total_path + 'player_friends_new.json'
        log_path = total_path + 'player_log.json'
        consume_path = total_path + 'player_consume.json'
        pvp_path = total_path + 'player_pvp.json'
        chat_path = total_path + 'player_chat.json'
        paid_path = total_path + 'player_paid.json'
        label_path = total_path + 'player_label.json'
        save_path = total_path + 'player_feature_label.json'
        logs = self.read_json(log_path)
        consumes = self.read_json(consume_path)
        pvps = self.read_json(pvp_path)
        chats = self.read_json(chat_path)
        paids = self.read_json(paid_path)
        labels = self.read_json(label_path)
        for filename in self._find_sub_file(attr_path):
            file_name = attr_path + '/' + filename
            with open(file_name, 'rb') as f:
                unix_time = int(time.mktime(time.strptime(filename.split('.')[0] + ' ' + '00:00:00', '%Y-%m-%d %H:%M:%S')))
                print(unix_time)
                # print(unix_time)
                for result in f:
                    temp_list = [unix_time]
                    result = result.decode('utf-8').strip('/').split()
                    if result[1] in ['17', '164', '230']:
                        if int(result[1]) == 17:
                            name = 's17'
                        elif int(result[1]) == 164:
                            name = 's164'
                        elif int(result[1]) == 230:
                            name = 's230'
                        profession = [8*[0]][0]
                        profession[int(result[4])-1] = 1
                        temp_list += profession
                        temp_list += list(map(float, result[5:]))
                        # login-logout
                        if result[0] in logs[name]['log']:
                            c_keys = list(map(int, list(logs[name]['log'][result[0]].keys())))
                            c_keys = list(map(str, [value for value in c_keys if value <= unix_time]))
                            if len(c_keys) != 0:
                                log_info = []
                                for key in c_keys:
                                    log_info += logs[name]['log'][result[0]][key]
                                temp_list.append(sum(log_info))
                            else:
                                temp_list.append(0)
                        else:
                            temp_list.append(0)
                        # pvp
                        if result[0] in pvps[name]['pvp']:
                            temp_list.append(len(list(filter(lambda x: x <= unix_time, pvps[name]['pvp'][result[0]]))))
                        else:
                            temp_list.append(0)
                        # chat
                        if result[0] in chats[name]['chat']:
                            temp_list.append(len(list(filter(lambda x: x <= unix_time, chats[name]['chat'][result[0]]))))
                        else:
                            temp_list.append(0)
                        # paid
                        if result[0] in paids[name]['paid']:
                            temp_list.append(len(list(filter(lambda x: x <= unix_time, paids[name]['paid'][result[0]]))))
                        else:
                            temp_list.append(0)
                        # spend（特指充值）
                        if result[0] in consumes[name]['consumption']:
                            c_keys = list(map(int, list(consumes[name]['consumption'][result[0]].keys())))
                            c_keys = list(map(str, [value for value in c_keys if value <= unix_time]))
                            if len(c_keys) != 0:
                                consume_info = []
                                for key in c_keys:
                                    consume_info += consumes[name]['consumption'][result[0]][key]
                                temp_list += [max(consume_info), round(sum(consume_info)/len(consume_info), 4), len(consume_info)]
                            else:
                                temp_list += [0, 0, 0]
                        else:
                            temp_list += [0, 0, 0]
                        # label
                        if result[0] in labels[name]['label']:
                            c_keys = list(map(int, list(labels[name]['label'][result[0]].keys())))
                            c_keys1 = list(map(str, [value for value in c_keys if value <= unix_time]))
                            if len(c_keys1) != 0:
                                _info = []
                                for key in c_keys1:
                                    _info += list(map(int, labels[name]['label'][result[0]][key]))
                                temp_list += [max(_info), round(sum(_info) / len(_info), 4), len(_info)]
                            else:
                                temp_list += [0, 0, 0]
                            if len(list(filter(lambda x: unix_time < x < unix_time + self.time_interval * 7, c_keys))) > 0:  # 30 days
                                temp_list.append(1)
                            else:
                                temp_list.append(0)
                        else:
                            temp_list += [0, 0, 0, 0]
                        # print("data:", temp_list)
                        self.data_dict[name].append(temp_list)

            print(file_name, "done!")

        self._save(save_path, self.data_dict)
        for key in self.data_dict.keys():
            np.save(total_path + 'instance_{}.npy'.format(key), np.array(self.data_dict[key]))

    def instance_check(self, ph):
        source_list = [17, 164, 230]
        for name in source_list:
            feature = ph + 'instance_s{}.npy'.format(name)
            data = np.load(feature)
            data.reshape([-1, 23])
            print(name, np.shape(data), list(data[:, -1]).count(1), list(data[:, -1]).count(1)/np.shape(data)[0])

    def analyze(self, path):
        path = path + 'player_label.json'
        result = self.read_json(path)
        stastic = {}
        keys = result.keys()
        for key in keys:
            stastic[key] = []
            for name in result[key]['id']:
                if name in result[key]['label']:
                    print(name, list(set(result[key]['label'][name])))
        #             for value in result[key]['label']:
        #                 stastic[key].append(len(set(result[key]['label'][value])))
        # for key in keys:
        #     print("{}_avg_day:".format(key), sum(stastic[key])/len(stastic[key]))
        #     print("{}_max_day:".format(key), max(stastic[key]))

    def reread_data(self, path, which):
        # np.set_printoptions(suppress=True)
        # path1 = path + 'instance_s{}.npy'.format(which)
        # data = np.load(path1)
        # data = data[np.argsort(data[:, 0])]
        # feature = data[:, :-1]
        # label = data[:, -1]
        # print(np.min(data[:, 0]), np.max(data[:, 0]))
        # print(np.min(feature[:, 0]), np.max(feature[:, 0]))
        # spilt_information = dict()
        # for i in range(feature.shape[1]-1):
        #     array = list(set(list(feature[:, i+1])))
        #     if len(array) > 10:
        #         spilt_information[i] = [i * 0.1 for i in range(0, 11)]
        #     else:
        #         spilt_information[i] = sorted(array)
        # print(spilt_information)
        # np.save(path + 'data_0.npy', feature)
        # np.save(path + 'label_0.npy', label)
        data_num = {}
        interval = 3600*24
        start_time = 1376582400
        for name in which:
            temp_num = []
            count = 0
            _path = path + 'instance_s{}.npy'.format(name)
            data = np.load(_path)
            data = data[np.argsort(data[:, 0])]
            for i, value in enumerate(data):
                if value[0] > start_time:
                    diff = int((value[0] - start_time) / interval)
                    for _ in range(diff):
                        start_time += interval
                        temp_num.append(count)
                        count = 0
                    count += 1
                else:
                    count += 1
            temp_num.append(count)
            start_time = 1376582400
            count = 0
            data_num[name] = temp_num
            print(data.shape[0])
            print(sum(temp_num[:4]))
            print(temp_num[:4])
            print('{}:avg, all, max, min'.format(name), sum(temp_num)/len(temp_num), sum(temp_num), max(temp_num), min(temp_num))
            if name == '17' or name == '164':
                num = 20000
            else:
                num = 16000
            # print(data[num:, 1:-1])
            feature = data[num:, 1:-1]
            # print(feature.shape)
            label = data[num:, -1]
            np.save(path + 'server{}/data_0.npy'.format(name), feature)
            np.save(path + 'server{}/label_0.npy'.format(name), label)

        # self._save(path + 'data_num.json', data_num)

    @staticmethod
    def read_in_chunks(filePath, chunk_size=1024 * 1024):
        file_object = open(filePath, 'r', encoding='utf-8')
        while True:
            chunk_data = file_object.read(chunk_size)
            if not chunk_data:
                break
            yield chunk_data

    @staticmethod
    def read_json(filePath):
        with open(filePath, 'r') as file:
            result = json.load(file)
        return result


if __name__ == "__main__":
    # servers = dict(s17=dict(id=set(), friendship=[], statistic=dict(maxNum=0, avg=0, count=[])), s164=dict(id=set(), friendship=[], statistic=dict(maxNum=0, avg=0, count=[])), s230=dict(id=set(), friendship=[], statistic=dict(maxNum=0, avg=0, count=[])))
    # ship_dict = dict()
    # save_path = 'C:/Users/徐懂事/Desktop/zj/netease/player_friend.json'
    # player_attr_path = 'E:/zju/netease/player_daily_attr'
    #
    # path_id = 'C:/Users/徐懂事/Desktop/zj/netease/id-account.txt'
    # friendship = ['C:/Users/徐懂事/Desktop/zj/netease/player_friend.txt.part1', 'C:/Users/徐懂事/Desktop/zj/netease/player_friend.txt.part2']
    # path_kejin = 'E:/zju/netease/chongzhi/chongzhi.txt.1'
    # path_mall = 'E:/zju/netease/mallshop/mallshop'
    path_test = 'E:/zju/netease/create_gender/player_create_time_gender(new).log'

    netease = NetEase()
    # np.get_player(player_attr_path, 'E:/zju/netease/processed/player_id.json')
    # np.get_friends('/home/fhz_11821062/czx_21821062/xds/processed/player_id.json', '/home/fhz_11821062/czx_21821062/xds/netease/friend', '/home/fhz_11821062/czx_21821062/xds/processed/player_friends_new.json')
    # np.get_consumption('/home/hdwu/xu/zju/netease/processed/player_id_friends.json', '/home/hdwu/xu/zju/netease/chongzhi', '/home/hdwu/xu/zju/netease/processed/player_ifc.json')
    # np.get_consumption('/home/fhz_11821062/czx_21821062/xds/processed/player_id.json', '/home/fhz_11821062/czx_21821062/xds/netease/chongzhi', '/home/fhz_11821062/czx_21821062/xds/processed/player_consume.json')
    # np.get_log('/home/fhz_11821062/czx_21821062/xds/processed/player_id.json', '/home/fhz_11821062/czx_21821062/xds/netease/login-logout', '/home/fhz_11821062/czx_21821062/xds/processed/player_log.json')
    # np.get_log('E:/zju/netease/processed/player_id.json', 'E:/zju/netease/login-logout', 'E:/zju/netease/processed/player_log.json')
    # np.get_label('E:/zju/netease/processed/player_id.json', 'E:/zju/netease/mallshop', 'E:/zju/netease/processed/player_label.json')
    # netease.get_label('/home/fhz_11821062/czx_21821062/xds/processed/player_id.json', '/home/fhz_11821062/czx_21821062/xds/netease/mallshop', '/home/fhz_11821062/czx_21821062/xds/processed/player_label.json')
    # netease.combine_features('E:/zju/netease/processed/')
    # netease.combine_features('/home/fhz_11821062/czx_21821062/xds/processed/')

    # netease.get_consumption('E:/zju/netease/processed/player_id.json', 'E:/zju/netease/chongzhi', 'E:/zju/netease/processed/player_consume.json')
    # netease.get_log('E:/zju/netease/processed/player_id.json', 'E:/zju/netease/login-logout', 'E:/zju/netease/processed/player_log.json')
    # netease.get_pvp('E:/zju/netease/processed/player_id.json', 'E:/zju/netease/player_kill_player', 'E:/zju/netease/processed/player_pvp.json')
    # netease.get_paid('E:/zju/netease/processed/player_id.json', 'E:/zju/netease/zengsong_daifu', 'E:/zju/netease/processed/player_paid.json')
    # netease.get_chat('E:/zju/netease/processed/player_id.json', 'E:/zju/netease/chat', 'E:/zju/netease/processed/player_chat.json')
    # netease.get_label('E:/zju/netease/processed/player_id.json', 'E:/zju/netease/shangcheng_goumai', 'E:/zju/netease/processed/player_label.json')
    # netease.combine_features('E:/zju/netease/processed/')

    # netease.instance_check('E:/zju/netease/processed/')
    # netease.instance_check('/home/fhz_11821062/czx_21821062/xds/processed/')
    # netease.analyze('E:/zju/netease/processed/')

    netease.reread_data('E:/zju/netease/processed/7/', ['17', '164', '230'])

    # with open(path_test, 'rb') as f:
    #     for result in f:
    #         result = result.decode('utf-8').strip('/').split()
    #         print(result)

