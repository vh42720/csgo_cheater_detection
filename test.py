"""
Temporary file for testing various functions and their results.
This file will be removed after project completion.
"""

import pickle
import pprint

from csgo_cheater_detection.config.config import data_path


with open(f'{data_path}\\score_dict.txt', 'rb') as fp:
	score_dict = pickle.load(fp)

pprint.pprint(score_dict)

