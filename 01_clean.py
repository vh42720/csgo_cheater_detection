"""
Batch info
----------
batch_date: August 30, 2020
number of parts: 5
number of steam IDs scraped: 50000

Combine multiple datasets into one and perform
some simple cleaning steps.
Each part of a batch contains the result of
scraping 10000 steam IDs.

Some descriptions on the data set:
	1. label: VAC ban status
	2. game_count: number of games a player owns
	3. playtime_forever: time played CS:GO
	4. all other columns are CS:GO statistics
	such as kills, deaths, accuracy, win, loses...
"""

import glob
import pandas as pd

from csgo_cheater_detection.config.config import *

# parameters
data_path = data_path
file_header = file_header
extension = extension
batch_date = batch_date

# combine data
all_filenames = [
	i for i in glob.glob(
		f'{data_path}\\{file_header}_{batch_date}_part_[0-9].{extension}'
	)
]

# combine all files in the list
df = pd.concat([pd.read_csv(f) for f in all_filenames])

# drop Unnamed: 0 column
df = df.drop('Unnamed: 0', axis=1)

# There is nearly 1.5 millions missing values
# Remove columns with 80%+ missing value.
# These values represents cut contents such as
# weapons and maps that did not get used by most
# players. Some cut contents are kept in the dataset
# as this signify long time players which is useful.
# Columns count goes from 289 to 219.
# Missing values goes from 1.5 mil to 510k.
thresh = 0.9
df = df.loc[:, df.isnull().sum() < thresh * df.shape[0]]

# The rest of the missing values stems from unpopular
# weapons, maps, or game objectives. It is reasonable
# to fill in these with 0.
df = df.fillna(0)

# For each weapon (29 in total), there is a shots and
# hits statistic. It might be beneficial to combine them
# as accuracy statistics. The special case is total hits
# and total shots which create total accuracy
# Remove both total shots hits and fired afterward
df['total_accuracy'] = df['total_shots_hit'] / df['total_shots_fired']
df = df.drop(['total_shots_hit', 'total_shots_fired'], axis=1)

# getting weapons statistics columns. The special column is
# total_shots_taser which has no hits recorded.
shots_cols = df.filter(regex='total_shots').columns\
	.drop('total_shots_taser')
hits_cols = df.filter(regex='total_hits').columns
weapons = hits_cols.str.replace('total_hits_', '')

# calculate weapon accuracy
for shot, hit, weapon in zip(shots_cols, hits_cols, weapons):
	# accuracy = hit / shots
	# pandas convert zero division to Nan
	col_name = f'accuracy_{weapon}'
	df[col_name] = df[hit] / df[shot]

	# remove the hit and shot column
	df = df.drop(shot, axis=1)
	df = df.drop(hit, axis=1)

	# refill the NaN values with 0
	# note that these are the only missing values NOW
	df = df.fillna(0)

# Before transforming even more statistics, we need to
# get rid of 100+ players with impossible statistics.
# Perhaps some of these players are testers.
# Ex: df.iloc[553,:] on 08-30-20 batch returns player
# with 10000 kills without deaths, no played time...
df = df[df['total_rounds_played'] > 0]
df = df[df['total_time_played'] > 0]
df = df[df['total_deaths'] > 0]

# Transform even more statistics to ratio
# [reference](https://nexosis.github.io/tutorials/detecting-cheaters-csgo)
df['win_ratio'] = df['total_wins'] / df['total_rounds_played']
df['kill_to_death_ratio'] = df['total_kills'] / df['total_deaths']
df['total_wins_per_hour'] = ((df['total_wins'] / df['total_time_played']) / 60) / 60
df['mvp_per_round'] = df['total_mvps'] / df['total_rounds_played']
df['total_headshots_per_round'] = df['total_kills_headshot'] / df['total_rounds_played']

# drop redundant columns
df = df.drop(
	['total_wins', 'total_rounds_played',
	 'total_kills', 'total_deaths',
	 'total_mvps', 'total_kills_headshot',
	 'total_time_played'], axis=1,
)

# Finally, drop 4 records with inf values
# Changing option to use infinite as nan
with pd.option_context('mode.use_inf_as_na', True):
	# Dropping the rows with nan (or inf) values
	df.dropna(inplace=True)

# export to csv
df.to_csv(f'{data_path}\\{file_header}_{batch_date}_full.csv',
          index=False,
          encoding='utf-8-sig')
