"""

"""

import pickle
import pandas as pd

from steam.webapi import WebAPI

from csgo_cheater_detection.utils.functions import *
from csgo_cheater_detection.config.config import *

# parameters
api_key = api_key
appid = appid
data_path = data_path

# loading list of steamids
with open(f'{data_path}\\steamids.txt', 'rb') as fp:
    steamids = pickle.load(fp)

samples = list(steamids)[0:4000]

# access
api = WebAPI(api_key)

# create many lists that store feature
game_count, playtime_forever, label, stats_list = [], [], [], []

for steamid in samples:
    # get player summaries
    status = get_player_summaries(
        api=api,
        steamid=steamid
    )

    # skip to the next steam id if the profile is not public
    code = status['response']['players'][0]['communityvisibilitystate']
    if code != 3:
        print(f'Private profile {steamid}')
        continue

    # get all games owned by player
    out = get_owned_games(
        api_key=api_key,
        steamid=steamid
    )

    # skip to the next steamID if response is empty
    if out['response'] == {}:
        print(f'Empty game record {steamid}')
        continue

    # record play time for CSGO (appid=730)
    try:
        game_list = out['response']['games']
    except KeyError:
        print(f'KeyError get_own_games {steamid}')
        continue

    # save playtime for CSGO
    playtime = next((
        game['playtime_forever']
        for game in game_list
        if game['appid'] == appid),
        0
    )

    # if playtime is 0, skip to the next steamid
    if playtime == 0:
        print(f'Does not own CSGO {steamid}')
        continue

    # save game count
    # append game_count and playtime_forever list
    game_count_temp = out['response']['game_count']

    # get VAC banned status of the player
    out = get_player_bans(
        api=api,
        steamid=steamid
    )

    # skip player who is banned before CSGO released
    if out['players'][0]['DaysSinceLastBan'] > time_since_csgo_release:
        print(f'Banned before released {steamid}')
        continue

    # save the label to the list
    label_temp = out['players'][0]['VACBanned']

    # get player statistics for CSGO
    out = get_user_stats(
        api=api,
        steamid=steamid,
        appid=appid
    )

    # get the list of player statistics.
    # skip if the list is somehow empty.
    try:
        player_stats = out['playerstats']['stats']
    except KeyError:
        print(f'KeyError get_user_stats {steamid}')
        continue

    # save the statistics for players
    stats_dict = {}
    for stats in player_stats:
        stats_dict[stats['name']] = stats['value']
    stats_list.append(stats_dict)

    # save the rest of the feature to list
    playtime_forever.append(playtime)
    game_count.append(game_count_temp)
    label.append(label_temp)

    # print retrieval status
    print('Success!')

# join together all lists to create one data frame
df = pd.DataFrame(stats_list)
df['game_count'] = game_count
df['playtime'] = playtime_forever
df['label'] = label
