"""
Initialize the steamids file that contains all steam ids pulled from
vacbanned.com.
"""

import pickle
from csgo_cheater_detection.config.config import data_path

# first list
steamids = {76561198000467276,
            76561198006920295,
            76561198097069831,
            76561198115776128,
            76561198163810747,
            76561198201999913,
            76561198295819486,
            76561198342663846,
            76561198385451528,
            76561198405098012,
            76561198434174544,
            76561198438666936,
            76561198441913401,
            76561198442778192,
            76561198625689154,
            76561198836739679,
            76561198857524018,
            76561198888349868,
            76561198934646619,
            76561198976662975,
            76561198982010111,
            76561199006812456,
            76561199029676147,
            76561199056988222,
            76561199067075216,
            76561199082508830}

# data path
data_path = data_path

# Create and save the initial file.
with open(f'{data_path}\\steamids.txt', "wb") as fp:
    pickle.dump(steamids, fp)
