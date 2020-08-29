"""
Combine multi-part data and clean up
"""

import glob
import pandas as pd

from csgo_cheater_detection.config.config import *


# parameters
data_path = data_path
file_header = 'csgo_cheater_data'
extension = 'csv'
patch_date = '8_26_20'

# combine data
all_filenames = [
	i for i in glob.glob(
		f'{data_path}\\{file_header}_{patch_date}_part_[0-9].{extension}'
	)
]

# combine all files in the list
df = pd.concat([pd.read_csv(f) for f in all_filenames])

# drop Unnamed: 0 column
df = df.drop('Unnamed: 0', axis=1)

# fill missing value with 0
df = df.fillna(0)

# export to csv
df.to_csv(f'{file_header}_{patch_date}_full.csv',
          index=False,
          encoding='utf-8-sig')
