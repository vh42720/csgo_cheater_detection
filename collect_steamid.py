"""

"""

from bs4 import BeautifulSoup
import requests
import pickle
import schedule
import time

from csgo_cheater_detection.config.config import last20_url, data_path

# parameters
last20_url = last20_url
data_path = data_path


# Define the job to run every 1 minute
def job():
	# load the set of banned steamids
	with open(f'{data_path}\\steamids.txt', "rb") as fp:
		steamids = pickle.load(fp)

	# request from vacbanned.com the newly checked 20 steamids
	r = requests.get(url=last20_url)
	soup = BeautifulSoup(r.text, 'html.parser')

	# adding new steamids pulled from the website
	for s in soup.select('table tr td:nth-of-type(2)'):
		steamids.add(int(s.get_text()))

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
