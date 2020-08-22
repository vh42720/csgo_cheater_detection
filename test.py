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

from csgo_cheater_detection.utils.functions import *
from csgo_cheater_detection.config.config import *

# parameters from config.py
api_key = api_key
appid = appid
data_path = data_path
time_since_csgo_release = time_since_csgo_release

# check if they update
steamids = {76561198000467276,
            76561198006920295,
            76561198097069831,
            76561198115776128,
            76561198163810747,
            76561198201999913,
            76561198295819486,
            76561198342663846,
            76561198385451528,
            76561198405098012}

# access API
api = WebAPI(api_key)

# game count list
game_count = []

# CSGO index list
playtime_forever = []

# label list used for identify cheaters
label = []

# stats list
stats_list = []


for steamid in steamids:

    # get player summaries
    status = get_player_summaries(
        api=api,
        steamid=steamid
    )

    # skip to the next steam id if the profile is not public
    code = status['response']['players'][0]['communityvisibilitystate']
    if code != 3:
        print('Private profile')
        continue

    # get all games owned by player
    out = get_owned_games(
        api_key=api_key,
        steamid=steamid
    )

    # skip to the next steamID if response is empty
    if out['response'] == {}:
        print('Empty game record')
        continue

    # record play time for CSGO (appid=730)
    game_list = out['response']['games']
    playtime = next((
        game['playtime_forever']
        for game in game_list
        if game['appid'] == appid),
        0
    )

    # if playtime is 0, skip to the next steamid
    if playtime == 0:
        print('Does not own CSGO')
        continue

    # append game_count and playtime_forever list
    game_count.append(out['response']['game_count'])
    playtime_forever.append(playtime)

    # get VAC banned status of the player
    out = get_player_bans(
        api=api,
        steamid=steamid
    )

    # skip player who is banned before CSGO released
    if out['players'][0]['DaysSinceLastBan'] > time_since_csgo_release:
        continue

    # save the label to the list
    label.append(out['players'][0]['VACBanned'])

    # get player statistics for CSGO
    out = get_user_stats(
        api=api,
        steamid=steamid,
        appid=appid
    )

    # save the stats
    stats_dict = {}
    for stats in out['playerstats']['stats']:
        stats_dict[stats['name']] = stats['value']
    stats_list.append(stats_dict)

    # print retrieval status
    print('Success!')


