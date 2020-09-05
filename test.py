"""
Temporary file for testing various functions and their results.
This file will be removed after project completion.
"""

import json
import pandas as pd
import requests
from bs4 import BeautifulSoup
from csgo_cheater_detection.config.config import *
from csgo_cheater_detection.utils.functions import *

# parameters
steamid = 76561198873575426

# run
steaminfo = {
        'key': api_key,
        'steamid': steamid,
        'format': 'JSON',
        'include_appinfo': '1'
    }
r = requests.get('http://api.steampowered.com/IPlayerService/GetOwnedGames/v0001/', params=steaminfo)

# if playtime
try:
	playtime = next((
	        game['playtime_2weeks']
	        for game in r.json()['response']['games']
	        if game['appid'] == 730),
	        0
	    )
except KeyError:
	playtime = 0

print(r.text)
