"""
  Dataset const
  Dataset contain: movie dataset
    movie dataset contains: ['movie_lens', 'netflix', 'rotten_tomatoes', 'tweet']
"""
import os


class Dataset(object):
    # DATASET = ['prsa_data', 'movie_data', 'netease_data']
    # DATASET = ['prsa_data']
    # DATASET = ['movie_data']
    DATASET = ['netease_data']

    # PATH = 'backend/data/'
    # PATH = 'D:/SSD_workspace/DaRtCase/backend/data/'
    PATH = 'E:/zju/movie_data'

    MOVIE_DATASET_NAME = 'movie_data'
    # MOVIE_SUB_DATASET = ['rotten_tomatoes', 'twitter', 'movie_lens']
    MOVIE_SUB_DATASET = ['MovieLens', 'RottenTomatoes', 'IMDB']
    # MOVIE_SUB_DATASET = ['movie_lens', 'netflix']
    MOVIE_SUB_DATASET_LENGTH = {'RottenTomatoes': 66398, 'IMDB': 316393, 'MovieLens': 1036496}
    MOVIE_SUB_DATASET_AVG_NUM = {'RottenTomatoes': 100, 'IMDB': 500, 'MovieLens': 1500}
    MOVIE_SUB_DATASET_DRIFT_INTERVAL = {'RottenTomatoes': 2, 'IMDB': 10, 'MovieLens': 30}
    MOVIE_SPILT_INFORMATION = {0: [0.0, 0.1, 0.2, 0.30000000000000004, 0.4, 0.5, 0.6000000000000001, 0.7000000000000001, 0.8, 0.9, 1.0], 1: [0.0, 0.1, 0.2, 0.30000000000000004, 0.4, 0.5, 0.6000000000000001, 0.7000000000000001, 0.8, 0.9, 1.0], 2: [0.0, 0.1, 0.2, 0.30000000000000004, 0.4, 0.5, 0.6000000000000001, 0.7000000000000001, 0.8, 0.9, 1.0], 3: [0.0, 0.1, 0.2, 0.30000000000000004, 0.4, 0.5, 0.6000000000000001, 0.7000000000000001, 0.8, 0.9, 1.0], 4: [0.0, 1.0], 5: [0.0, 1.0], 6: [0.0, 1.0], 7: [0.0, 1.0], 8: [0.0, 1.0], 9: [0.0, 1.0], 10: [0.0, 1.0], 11: [0.0, 1.0], 12: [0.0, 1.0], 13: [0.0, 1.0], 14: [0.0, 1.0], 15: [0.0, 1.0]}
    MOVIE_ATTRIBUTES = [('year', 1), ('duration', 1), ('budget', 1), ('direction', 1), ('genre', 10), ('language', 2)]
    MOVIE_DIM = 16

    PRSA_DATASET_NAME = 'prsa_data'
    PRSA_SUB_DATASET = ['Guanyuan', 'Tiantan', 'Wanshouxigong', 'Dongsi']
    PRSA_SUB_DATASET_LENGTH = {'Guanyuan': 35040, 'Tiantan': 35040, 'Wanshouxigong': 35040, 'Dongsi': 35040}
    PRSA_SUB_DATASET_AVG_NUM = {'Guanyuan': 24, 'Tiantan': 24, 'Wanshouxigong': 24, 'Dongsi': 24}
    PRSA_SUB_DATASET_DRIFT_INTERVAL = {'Guanyuan': 1, 'Tiantan': 1, 'Wanshouxigong': 1, 'Dongsi': 1}
    PRSA_SPILT_INFORMATION = {0: [0.0, 0.25, 0.5, 0.75, 1.0], 1: [0.0, 0.1, 0.2, 0.30000000000000004, 0.4, 0.5, 0.6000000000000001, 0.7000000000000001, 0.8, 0.9, 1.0], 2: [0.0, 0.1, 0.2, 0.30000000000000004, 0.4, 0.5, 0.6000000000000001, 0.7000000000000001, 0.8, 0.9, 1.0], 3: [0.0, 0.1, 0.2, 0.30000000000000004, 0.4, 0.5, 0.6000000000000001, 0.7000000000000001, 0.8, 0.9, 1.0], 4: [0.0, 0.1, 0.2, 0.30000000000000004, 0.4, 0.5, 0.6000000000000001, 0.7000000000000001, 0.8, 0.9, 1.0], 5: [0.0, 0.1, 0.2, 0.30000000000000004, 0.4, 0.5, 0.6000000000000001, 0.7000000000000001, 0.8, 0.9, 1.0], 6: [0.0, 0.1, 0.2, 0.30000000000000004, 0.4, 0.5, 0.6000000000000001, 0.7000000000000001, 0.8, 0.9, 1.0], 7: [0.0, 0.1, 0.2, 0.30000000000000004, 0.4, 0.5, 0.6000000000000001, 0.7000000000000001, 0.8, 0.9, 1.0], 8: [0.0, 0.1, 0.2, 0.30000000000000004, 0.4, 0.5, 0.6000000000000001, 0.7000000000000001, 0.8, 0.9, 1.0], 9: [0.0, 0.1, 0.2, 0.30000000000000004, 0.4, 0.5, 0.6000000000000001, 0.7000000000000001, 0.8, 0.9, 1.0], 10: [0.0, 0.1, 0.2, 0.30000000000000004, 0.4, 0.5, 0.6000000000000001, 0.7000000000000001, 0.8, 0.9, 1.0], 11: [0.0, 0.1, 0.2, 0.30000000000000004, 0.4, 0.5, 0.6000000000000001, 0.7000000000000001, 0.8, 0.9, 1.0], 12: [0.0, 0.1, 0.2, 0.30000000000000004, 0.4, 0.5, 0.6000000000000001, 0.7000000000000001, 0.8, 0.9, 1.0], 13: [0.0, 0.1, 0.2, 0.30000000000000004, 0.4, 0.5, 0.6000000000000001, 0.7000000000000001, 0.8, 0.9, 1.0], 14: [0.0, 0.1, 0.2, 0.30000000000000004, 0.4, 0.5, 0.6000000000000001, 0.7000000000000001, 0.8, 0.9, 1.0], 15: [0.0, 0.1, 0.2, 0.30000000000000004, 0.4, 0.5, 0.6000000000000001, 0.7000000000000001, 0.8, 0.9, 1.0], 16: [0.0, 0.1, 0.2, 0.30000000000000004, 0.4, 0.5, 0.6000000000000001, 0.7000000000000001, 0.8, 0.9, 1.0], 17: [0.0, 0.1, 0.2, 0.30000000000000004, 0.4, 0.5, 0.6000000000000001, 0.7000000000000001, 0.8, 0.9, 1.0], 18: [0.0, 0.1666666716337204, 0.25, 0.3333333432674408, 0.5], 19: [0.0, 0.1666666716337204, 0.25, 0.3333333432674408, 0.5], 20: [0.0, 0.1666666716337204, 0.25, 0.3333333432674408, 0.5], 21: [0.0, 0.1666666716337204, 0.25, 0.3333333432674408, 0.5]}
    PRSA_ATTRIBUTES = [('year', 1), ('month', 1), ('day', 1), ('hour', 1), ('SO2', 1), ('NO2', 1), ('CO', 1), ('O3', 1), ('O3_8hours', 1), ('PM2.5', 1), ('PM2.5_day', 1), ('PM10', 1), ('PM10_day', 1), ('TEMP', 1), ('PRES', 1), ('DEWP', 1), ('RAIN', 1), ('WSPM', 1), ('wd', 4)]
    PRSA_DIM = 22

    NETEASE_DATASET_NAME = 'netease_data'
    NETEASE_SUB_DATASET = ['server17', 'server164', 'server230']
    NETEASE_SUB_DATASET_LENGTH = {'server17': 667800, 'server164': 722125, 'server230': 601048}
    NETEASE_SUB_DATASET_AVG_NUM = {'server17': 5000, 'server164': 5000, 'server230': 4000}
    NETEASE_SUB_DATASET_DRIFT_INTERVAL = {'server17': 200, 'server164': 200, 'server230': 160}
    NETEASE_SPILT_INFORMATION = {0: [0.0, 1.0], 1: [0.0, 1.0], 2: [0.0, 1.0], 3: [0.0, 1.0], 4: [0.0, 1.0], 5: [0.0, 1.0], 6: [0.0, 1.0], 7: [0.0, 1.0], 8: [0.0, 0.1, 0.2, 0.30000000000000004, 0.4, 0.5, 0.6000000000000001, 0.7000000000000001, 0.8, 0.9, 1.0], 9: [0.0, 0.1, 0.2, 0.30000000000000004, 0.4, 0.5, 0.6000000000000001, 0.7000000000000001, 0.8, 0.9, 1.0], 10: [0.0, 0.1, 0.2, 0.30000000000000004, 0.4, 0.5, 0.6000000000000001, 0.7000000000000001, 0.8, 0.9, 1.0], 11: [0.0, 0.1, 0.2, 0.30000000000000004, 0.4, 0.5, 0.6000000000000001, 0.7000000000000001, 0.8, 0.9, 1.0], 12: [0.0, 0.1, 0.2, 0.30000000000000004, 0.4, 0.5, 0.6000000000000001, 0.7000000000000001, 0.8, 0.9, 1.0], 13: [0.0, 0.1, 0.2, 0.30000000000000004, 0.4, 0.5, 0.6000000000000001, 0.7000000000000001, 0.8, 0.9, 1.0], 14: [0.0, 0.1, 0.2, 0.30000000000000004, 0.4, 0.5, 0.6000000000000001, 0.7000000000000001, 0.8, 0.9, 1.0], 15: [0.0, 0.1, 0.2, 0.30000000000000004, 0.4, 0.5, 0.6000000000000001, 0.7000000000000001, 0.8, 0.9, 1.0], 16: [0.0, 0.1, 0.2, 0.30000000000000004, 0.4, 0.5, 0.6000000000000001, 0.7000000000000001, 0.8, 0.9, 1.0], 17: [0.0, 0.1, 0.2, 0.30000000000000004, 0.4, 0.5, 0.6000000000000001, 0.7000000000000001, 0.8, 0.9, 1.0], 18: [0.0, 0.1, 0.2, 0.30000000000000004, 0.4, 0.5, 0.6000000000000001, 0.7000000000000001, 0.8, 0.9, 1.0], 19: [0.0, 0.1, 0.2, 0.30000000000000004, 0.4, 0.5, 0.6000000000000001, 0.7000000000000001, 0.8, 0.9, 1.0], 20: [0.0, 0.1, 0.2, 0.30000000000000004, 0.4, 0.5, 0.6000000000000001, 0.7000000000000001, 0.8, 0.9, 1.0]}
    NETEASE_ATTRIBUTES = [('profession', 8), ('equipment', 1), ('practice', 1), ('mastery', 1), ('online_time', 1), ('max_recharge', 1), ('ave_recharge', 1),
                          ('count_recharge', 1), ('count_pvp', 1), ('count_chat', 1), ('count_gift', 1), ('max_purchase', 1), ('ave_purchase', 1),
                          ('count_purchase', 1)]
    NETEASE_DIM = 21

    @staticmethod
    def get_path(dataset, sub_dataset):
        path = '{}/{}/{}/'.format(Dataset.PATH, dataset, sub_dataset)
        if not os.path.exists(path):
            os.mkdir(path)
        return path

    @staticmethod
    def get_sub_dataset(dataset):
        if dataset == Dataset.MOVIE_DATASET_NAME:
            return Dataset.MOVIE_SUB_DATASET
        elif dataset == Dataset.PRSA_DATASET_NAME:
            return Dataset.PRSA_SUB_DATASET
        elif dataset == Dataset.NETEASE_DATASET_NAME:
            return Dataset.NETEASE_SUB_DATASET

    @staticmethod
    def get_attributes(dataset):
        if dataset == Dataset.MOVIE_DATASET_NAME:
            return Dataset.MOVIE_ATTRIBUTES
        elif dataset == Dataset.PRSA_DATASET_NAME:
            return Dataset.PRSA_ATTRIBUTES
        elif dataset == Dataset.NETEASE_DATASET_NAME:
            return Dataset.NETEASE_ATTRIBUTES

    @ staticmethod
    def get_length(dataset, sub_dataset):
        if dataset == Dataset.MOVIE_DATASET_NAME:
            return Dataset.MOVIE_SUB_DATASET_LENGTH[sub_dataset]
        elif dataset == Dataset.PRSA_DATASET_NAME:
            return Dataset.PRSA_SUB_DATASET_LENGTH[sub_dataset]
        elif dataset == Dataset.NETEASE_DATASET_NAME:
            return Dataset.NETEASE_SUB_DATASET_LENGTH[sub_dataset]

    @staticmethod
    def get_spilt_inform(dataset):
        if dataset == Dataset.MOVIE_DATASET_NAME:
            return Dataset.MOVIE_SPILT_INFORMATION
        elif dataset == Dataset.PRSA_DATASET_NAME:
            return Dataset.PRSA_SPILT_INFORMATION
        elif dataset == Dataset.NETEASE_DATASET_NAME:
            return Dataset.NETEASE_SPILT_INFORMATION

    @staticmethod
    def get_detect_interval(dataset, sub_dataset):
        if dataset == Dataset.MOVIE_DATASET_NAME:
            return Dataset.MOVIE_SUB_DATASET_DRIFT_INTERVAL[sub_dataset]
        elif dataset == Dataset.PRSA_DATASET_NAME:
            return Dataset.PRSA_SUB_DATASET_DRIFT_INTERVAL[sub_dataset]
        elif dataset == Dataset.NETEASE_DATASET_NAME:
            return Dataset.NETEASE_SUB_DATASET_DRIFT_INTERVAL[sub_dataset]

    @staticmethod
    def get_detect_batch_interval(dataset, sub_dataset):
        if dataset == Dataset.MOVIE_DATASET_NAME:
            return Dataset.MOVIE_SUB_DATASET_AVG_NUM[sub_dataset] * 4
        elif dataset == Dataset.PRSA_DATASET_NAME:
            return 100
        elif dataset == Dataset.NETEASE_DATASET_NAME:
            return Dataset.NETEASE_SUB_DATASET_AVG_NUM[sub_dataset] * 4

    @staticmethod
    def get_online_learning_batch_interval(dataset, sub_dataset):
        if dataset == Dataset.MOVIE_DATASET_NAME:
            return Dataset.MOVIE_SUB_DATASET_AVG_NUM[sub_dataset] * 4
        elif dataset == Dataset.PRSA_DATASET_NAME:
            return 500
        elif dataset == Dataset.NETEASE_DATASET_NAME:
            return Dataset.NETEASE_SUB_DATASET_AVG_NUM[sub_dataset] * 4
