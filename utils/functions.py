"""
Contains all the functions used in the projects.
"""

import requests
import json


def get_player_summaries(api, steamid):
    """Retrieve basic profile information including communityvisibilitystate.

    As long as the communityvisibilitystate is set to 3,
    in-game statistics can be retrieved.
    communityvisibilitystate:
        1 - Private/FriendsOnly, player statistics is hidden.
        3 - Public, player statistics is visible.

    The function will return the error message if the call failed.
    This can happen if the profile is deleted.

    :param api: steam API instance from steam.webapi.WebAPI
    :param steamid: integer, steam id of a user.
    :return: out: a dictionary of basic profile information

    References
    ----------
    https://steam.readthedocs.io/en/stable/api/steam.webapi.html
    https://developer.valvesoftware.com/wiki/Steam_Web_API#GetPlayerSummaries_.28v0002.29
    """
    out = ''
    try:
        out = api.call(
            'ISteamUser.GetPlayerSummaries',
            steamids=steamid
        )
    except requests.exceptions.HTTPError as e:
        print('Error when GetPlayerSummaries', e)

    return out


def get_owned_games(api_key, steamid):
    """Retrieve all the games owned and playtime by player

    Sometimes the profile is empty (without any games).
    Number of games owned by the player is saved to a list
    and will be used a a feature for prediction.
    The only parameter that is useful for CSGO is playtime.

    :param api_key: string, key for steam API
    :param steamid: integer, steam id of a user.
    :return: out: dictionary of playtime.
    If the game library is empty, return an empty dict.

    Reference
    ---------
    https://steam.readthedocs.io/en/stable/api/steam.webapi.html
    https://developer.valvesoftware.com/wiki/Steam_Web_API#GetOwnedGames_.28v0001.29
    """
    steaminfo = {
        'key': api_key,
        'steamid': steamid,
        'format': 'JSON',
        'include_appinfo': '1'
    }
    out = requests.get('http://api.steampowered.com/IPlayerService/GetOwnedGames/v0001/', params=steaminfo)
    out.close()

    # there is a chance the conversion fail
    # if that happens, return an empty dict
    try:
        return out.json()
    except json.decoder.JSONDecodeError:
        return {}


def get_player_bans(api, steamid):
    """Retrieve all recorded VAC banned for the player.

    VAC ban is not permanent and thus each user can have
    multiple instances of bans. VAC ban will be the label
    column with True for at least one ban and False for none.

    :param api: steam API instance from steam.webapi.WebAPI
    :param steamid: integer, steam id of a user.
    :return: out:

    Reference
    ---------
    https://steam.readthedocs.io/en/stable/api/steam.webapi.html
    https://developer.valvesoftware.com/wiki/Steam_Web_API#GetPlayerBans_.28v1.29
    """
    out = ''
    try:
        out = api.call(
            'ISteamUser.GetPlayerBans',
            steamids=steamid
        )
    except requests.exceptions.HTTPError as e:
        print('Error when GetPlayerBans', e)

    return out


def get_user_stats(api, steamid, appid):
    """Retrieve user for statistics for a specific steam game.

    The aforementioned statistics is different for each game.

    :param api: steam API instance from steam.webapi.WebAPI
    :param steamid: integer, steam id of a user.
    :param appid: integer, the id of the game.
    :return: out: a dictionary contains all the user statistics
    of a game played by user.

    Reference
    ---------
    https://steam.readthedocs.io/en/stable/api/steam.webapi.html
    https://developer.valvesoftware.com/wiki/Steam_Web_API#GetUserStatsForGame_.28v0002.29
    """
    out = ''
    try:
        out = api.call(
            'ISteamUserStats.GetUserStatsForGame',
            steamid=steamid,
            appid=appid
        )
    except requests.exceptions.HTTPError as e:
        print('Error with interface GetUserStatsForGame', e)

    return out
