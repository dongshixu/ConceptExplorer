"""
  Dataset const
  Dataset contain: movie dataset
    movie dataset contains: ['movie_lens', 'netflix', 'rotten_tomatoes', 'tweet']
"""
import os


class Dataset(object):
	# DATASET = ['prsa_data', 'movie_data']
	# DATASET = ['prsa_data']
	DATASET = ['movie_data']

	# PATH = 'D:/SSD_workspace/DaRtCase/backend/data/'
	PATH = 'E:/zju/movie_data'

	# PATH = 'backend/data/'

	MOVIE_DATASET_NAME = 'movie_data'
	# MOVIE_SUB_DATASET = ['rotten_tomatoes', 'twitter', 'movie_lens', 'netflix']
	MOVIE_SUB_DATASET = ['movie_lens', 'netflix']
	MOVIE_ATTRIBUTES = [('year', 1), ('duration', 1), ('budget', 1), ('direction', 1), ('genre', 10), ('language', 2), ('review_date', 1)]
	MOVIE_DIM = 17

	PRSA_DATASET_NAME = 'prsa_data'
	PRSA_SUB_DATASET = ['Guanyuan', 'Tiantan', 'Wanshouxigong']
	PRSA_ATTRIBUTES = [('year', 1), ('month', 1), ('day', 1), ('hour', 1), ('SO2', 1), ('NO2', 1), ('CO', 1), ('O3', 1), ('O3_8hours', 1), ('PM2.5', 1), ('PM2.5_day', 1), ('PM10', 1), ('PM10_day', 1), ('TEMP', 1), ('PRES', 1), ('DEWP', 1), ('RAIN', 1), ('WSPM', 1), ('wd', 4)]
	PRSA_DIM = 22

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

	@staticmethod
	def get_attributes(dataset):
		if dataset == Dataset.MOVIE_DATASET_NAME:
			return Dataset.MOVIE_ATTRIBUTES
		elif dataset == Dataset.PRSA_DATASET_NAME:
			return Dataset.PRSA_ATTRIBUTES

