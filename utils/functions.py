"""
Contains all the functions used in the projects.
"""

import requests
import json
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

import keras
import tensorflow as tf


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
    :return: out: a dictionary contains VAC record.

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


def split_X_y(df, label_col, test_size, random_state):
    """Split dataframe into train set, validation set
    and test set with X and y for each:
           X_train, X_val, X_test
           y_train, y_val, y_test

    :param df: dataframe, original dataframe.
    :param label_col: string, name of the label column.
    :param test_size: float, percentage of test set.
    :param random_state: integer, number for reproducible.
    :return: 6 dataframes contain the train, validation and test
    sets.
    """
    # Split and shuffle datasets
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state
    )
    train_df, val_df = train_test_split(
        train_df,
        test_size=test_size,
        random_state=random_state
    )

    # Form y for each set
    y_train = train_df.pop(label_col)
    y_val = val_df.pop(label_col)
    y_test = test_df.pop(label_col)

    # Form X for each set
    X_train, X_val, X_test = train_df, val_df, test_df

    # standardize training data and transform test data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # return values in correct order
    return X_train, X_val, X_test, y_train, y_val, y_test


def best_random_forest_clf(param_grid, X_train, y_train):
    """

    :param param_grid:
    :param X_train:
    :param y_train:
    :return:
    """
    # Initiate random forest
    rf = RandomForestClassifier()

    # Run grid search model
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=3,
        n_jobs=-1,
        verbose=0)
    grid_search.fit(X_train, y_train)

    # refit with the best features
    rf = RandomForestClassifier(**grid_search.best_params_)
    rf.fit(X_train, y_train)

    return rf


def make_neural_network(input_dim=None, output_bias=None):
    """
    Define a model with 5 dense layers, a dropout layer to
    reduce over-fitting and a sigmoid output node that returns
    probability of a player being a cheater.

    :param input_dim: int, dimension of the input layer.
    :param output_bias: float, initial bias.
    :return: neural network ready for training.
    """

    # Define metrics and model configs
    METRICS = [
        keras.metrics.BinaryAccuracy(name='accuracy'),
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall'),
        keras.metrics.AUC(name='auc'),
    ]

    # configs for neural network
    layer_1_nodes = 190
    layer_2_nodes = 190
    layer_3_nodes = 190
    layer_4_nodes = 190
    layer_5_nodes = 95
    output_nodes = 1

    # configs for metrics and initial bias
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)

    # neural network model structure
    model = keras.Sequential([
        keras.layers.Dense(layer_1_nodes, activation='relu', input_dim=input_dim),
        keras.layers.Dense(layer_2_nodes, activation='relu'),
        keras.layers.Dense(layer_3_nodes, activation='relu'),
        keras.layers.Dense(layer_4_nodes, activation='relu'),
        keras.layers.Dense(layer_5_nodes, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(output_nodes, activation='sigmoid', bias_initializer=output_bias),
    ])

    # compile neural network
    model.compile(
        optimizer=keras.optimizers.Adam(lr=1e-3),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=METRICS
    )

    return model


def plot_roc(name, labels, predictions, **kwargs):
    """Plot the ROC curve for a given classifier

    :param name: string, the classifier name
    :param labels: array, the true y values
    :param predictions: array, the predicted y values
    :param kwargs: all other parameters if applicable
    :return: plot the curve and label its name accordingly

    Reference
    ---------
    https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#plot_the_roc
    """
    # calculate fp and tp
    fp, tp, _ = roc_curve(labels, predictions)

    # plot the curve
    plt.plot(100*fp, 100*tp, label=name, linewidth=2, **kwargs)
    plt.xlabel('False positives [%]')
    plt.ylabel('True positives [%]')
    # plt.xlim([-0.5,20])
    # plt.ylim([80,100.5])
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect('equal')


def plot_precision_recall(name, labels, predictions, **kwargs):
    """Plot the precision-recall curve for a given classifier

    :param name: string, the classifier name
    :param labels: array, the true y values
    :param predictions: array, the predicted y values
    :param kwargs: all other parameters if applicable
    :return: plot the curve and label its name accordingly

    Reference
    ---------
    https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/
    """
    # calculate precision recall values
    precision, recall, _ = precision_recall_curve(labels, predictions)

    # plot the curve
    plt.plot(recall, precision, label=name, linewidth=2, **kwargs)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect('equal')
