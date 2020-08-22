"""
Contains all the functions used in the projects.
"""

import requests


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
    :param steamid: steam id of a user.
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
    """

    :param api_key:
    :param steamid:
    :return: out:
    """
    steaminfo = {
        'key': api_key,
        'steamid': steamid,
        'format': 'JSON',
        'include_appinfo': '1'
    }
    out = requests.get('http://api.steampowered.com/IPlayerService/GetOwnedGames/v0001/', params=steaminfo)
    return out.json()


def get_player_bans(api, steamid):
    """
    
    :param api:
    :param steamid:
    :return: out:
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

    :param api:
    :param steamid:
    :param appid:
    :return: out:
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
