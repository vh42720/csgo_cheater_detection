"""
This file contains all configurations needed to run the scraping script.
For replication of the project, provide the the following:
	api_key: string, Steam Web API key. (https://steamcommunity.com/dev/)
	data_path: string, path to directory that contains all data sets of the project

Do not change other parameters!
"""

import datetime

# API keys
api_key = '7E9EF075ABE09DD48DFC8BD6E6CFAEE7'

# data path
data_path = 'D:\\PycharmProjects\\my_projects\\csgo_cheater_detection\\data'

# app id for CSGO on steam
appid = 730

# url for last 20 steam ids that get checked from vacbanned.com
last20_url = "http://www.vacbanned.com/listing/last"

# CSGO steam release date
csgo_release_date = datetime.datetime(2012, 8, 21)
now = datetime.datetime.now()
time_since_csgo_release = int((now - csgo_release_date).days)
