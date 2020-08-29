"""
Temporary file for testing various functions and their results.
This file will be removed after project completion.
"""

import pickle
import json
import pandas as pd
from csgo_cheater_detection.config.config import data_path
from csgo_cheater_detection.utils.functions import *
from steam.webapi import WebAPI

from tqdm import tqdm

from csgo_cheater_detection.utils.functions import *
from csgo_cheater_detection.config.config import *

# parameters from config.py
api_key = api_key
appid = appid
data_path = data_path
time_since_csgo_release = time_since_csgo_release

with open(f'{data_path}\\steamids.txt', 'rb') as fp:
    steamids = pickle.load(fp)

steamids = list(steamids)[36020:36050]

# access
api = WebAPI(api_key)

for steamid in tqdm(steamids):
    out = get_player_summaries(
        api=api,
        steamid=steamid
    )
    print(out['response']['players'])
