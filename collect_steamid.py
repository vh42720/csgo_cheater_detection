"""
Collects any steam IDs that are entered in the vacbanned.com website.
For each pass, the last 20 steam IDs are added. If the steam ID is
already in our set, it will be ignored.
The script is set to run every minute until stopped manually.

Reference
---------
http://www.vacbanned.com/
http://www.vacbanned.com/listing/last
"""

from bs4 import BeautifulSoup
import requests
import schedule
import pickle
import time

from csgo_cheater_detection.config.config import *

# parameters
vacbanned_last20 = vacbanned_last20
vaclist_last20 = vaclist_last20
data_path = data_path


# Define the job to run e very 1 minute
def job():
	# load the set of banned steamids
	with open(f'{data_path}\\steamids.txt', "rb") as fp:
		steamids = pickle.load(fp)

	# request from vacbanned.com the newly checked 20 steamids
	r = requests.get(url=vacbanned_last20)
	soup = BeautifulSoup(r.text, 'html.parser')
	r.close()

	# adding new steamids pulled from vacbanned
	# the steamids reside in the second column
	for s in soup.select('table tr td:nth-of-type(2)'):
		steamids.add(int(s.get_text()))

	# request from VacList.com the last 20 banned steamids
	r = requests.get(url=vaclist_last20)
	r.close()
	player_list = r.json()

	# adding new steamids pulled from vaclist
	for player in player_list:
		steamids.add(int(player['steam_id']))

	# print the number of steamids collected
	print(len(steamids))

	# Save the set after read and add new steamids
	with open(f'{data_path}\\steamids.txt', "wb") as fp:
		pickle.dump(steamids, fp)


# scheduling the job to run every minute
schedule.every(1).minutes.do(job)
schedule.every().hour.do(job)

while True:
	schedule.run_pending()
	time.sleep(1)
