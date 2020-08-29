"""
Collect player statistics for the list of steam IDs.
The steps to collect data with steam APIs:
    1. call GetPlayerSummaries - private profile is skipped
        a) communityvisibilitystate = 1/2: Private profile
        b) communityvisibilitystate = 3: Public profile
    2. call GetOwnedGames - account with no game is skipped
    3. call GetPlayerBans - record cheat/no cheat label
    4. call GetUserStatsForGame - record player statistic for games

If a steam ID passed through all checks, record the successful
and print on console.

Reference
---------
https://developer.valvesoftware.com/wiki/Steam_Web_API#GetPlayerSummaries_.28v0002.29
https://developer.valvesoftware.com/wiki/Steam_Web_API#GetOwnedGames_.28v0001.29
https://developer.valvesoftware.com/wiki/Steam_Web_API#GetPlayerBans_.28v1.29
https://api.steampowered.com/ISteamUserStats/GetUserStatsForGame/v0002/
"""

import pickle
import pandas as pd

from steam.webapi import WebAPI
from tqdm import tqdm

from csgo_cheater_detection.utils.functions import *
from csgo_cheater_detection.config.config import *

# parameters
api_key = api_key
appid = appid
data_path = data_path
public_code = 3
version = '_8_26_20_part_4'

# loading list of steamids
with open(f'{data_path}\\steamids.txt', 'rb') as fp:
    steamids = pickle.load(fp)

steamids = list(steamids)[30000:40000]

# access
api = WebAPI(api_key)

# create many lists that store feature
game_count, playtime_forever, label, stats_list = [], [], [], []

# loop through the steamids list
for steamid in tqdm(steamids):
    # get player account summaries and status
    summary = get_player_summaries(
        api=api,
        steamid=steamid
    )
    status = summary['response']['players']

    # skip to the next steam id if the profile is empty
    try:
        code = status[0]['communityvisibilitystate']
    except IndexError:
        continue

    # skip to the next steam id if the profile is not public
    if code != public_code:
        continue
    else:
        # get all games owned by player if profile is public
        out = get_owned_games(
            api_key=api_key,
            steamid=steamid
        )

    # skip to the next steamID if Json conversion fail
    if out == {}:
        continue
    # skip to the next steamID if response is empty
    elif out['response'] == {}:
        continue
    else:
        # record play time for CSGO (appid=730)
        try:
            game_list = out['response']['games']
        except KeyError:
            continue

    # save playtime for CSGO
    playtime = next((
        game['playtime_forever']
        for game in game_list
        if game['appid'] == appid),
        0
    )

    # if CSGO playtime is 0, skip to the next steamid
    if playtime == 0:
        continue
    else:
        # record the number of games player owned
        game_count_temp = out['response']['game_count']
        # get VAC banned status of the player
        out = get_player_bans(
            api=api,
            steamid=steamid
        )

    # skip player who is banned before CSGO released
    if out['players'][0]['DaysSinceLastBan'] > time_since_csgo_release:
        continue
    else:
        # save the label to the list
        label_temp = out['players'][0]['VACBanned']

        # get player statistics for CSGO
        out = get_user_stats(
            api=api,
            steamid=steamid,
            appid=appid
        )

    # get the list of player statistics.
    # skip the steam id if the list is somehow empty.
    try:
        player_stats = out['playerstats']['stats']
    except KeyError:
        continue

    # save the statistics for players
    stats_dict = {}
    for stats in player_stats:
        stats_dict[stats['name']] = stats['value']
    stats_list.append(stats_dict)

    # save the rest of the feature to their list
    playtime_forever.append(playtime)
    game_count.append(game_count_temp)
    label.append(label_temp)

# join together all lists to create one data frame
df = pd.DataFrame(stats_list)
df['game_count'] = game_count
df['playtime'] = playtime_forever
df['label'] = label

# save the data
df.to_csv(f'{data_path}\\csgo_cheater_data{version}.csv')
