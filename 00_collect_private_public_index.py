"""
***
Run this script only on steam special APIs so
the server won't cut connection. Else, run it in parts!
***

Collect status of steamids: public or private
This function reduce number of calls to steam server.
Steams restricts API  to 100000 calls per day.
And for each steamids, 1 to 4 calls needed to
collect data. It is impossible to collect data
for a list of 50k steamids in one day. It is very
error prone and hits the limit quickly. However,
if private profiles are removed a day before,
each day a full data set can be scraped.

This script will create the index of which steamid
is private and remove them before collect_data script.
private_public_index:
	- Private = False
	- Public = True
Reference
---------
https://steamcommunity.com/dev/apiterms
"""

import pickle

from collections import Counter
from steam.webapi import WebAPI
from tqdm import tqdm

from csgo_cheater_detection.config.config import \
	data_path, public_code, api_key
from csgo_cheater_detection.utils.functions import get_player_summaries

# parameters
data_path = data_path

# loading list of steamids
with open(f'{data_path}\\steamids.txt', 'rb') as fp:
	steamids = pickle.load(fp)

# convert a set to list for iteration
steamids = list(steamids)

# access
api = WebAPI(api_key)

# initiate index list
private_public_index = []

# loops to record status index
for steamid in tqdm(steamids):
	# get player account summaries and status
	summary = get_player_summaries(
		api=api,
		steamid=steamid
	)
	status = summary['response']['players']

	# skip to the next steam id if the profile is empty
	# record as false to remove when collecting data
	try:
		code = status[0]['communityvisibilitystate']
	except IndexError:
		private_public_index.append(False)
		continue

	# record public as True, private as False
	if code == public_code:
		private_public_index.append(True)
	else:
		private_public_index.append(False)

# Print number of private and public
print(Counter(private_public_index))

# save the list with pickle
with open(f'{data_path}\\private_public_index.txt', 'wb') as fp:
	pickle.dump(private_public_index, fp)
