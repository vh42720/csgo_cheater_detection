"""
Batch info
----------
batch_date: August 30, 2020
number of parts: 5
number of steam IDs scraped: 50000

Why resampling?
---------------
Imbalanced datasets contains severely skewed distribution
of labels such as small 2% fraud vs 98% non-fraud. This bias
can hinder many machine learning models. For the analogy,
lets assume {'True': 2, 'False': 98}.
One way to deal with imbalanced data is through resampling
methods: under-sampling and over-sampling.
	* under-sampling: randomly pick majority class up to
	the number of minority class. Results {'True': 2, 'False': 2}
	* over-sampling: randomly sample replacement of the
	minority class. Results {'True': 98, 'False': 98}

SMOTE and NearMiss are the more sophisticated methods of
over-sampling and under-sampling respectively.

Goal
----
Produce 4 separate dataset through resampling for fitting:
	1. Random over-sampling
	2. SMOTE
	3. Random under-sampling
	4. Near Miss

Reference
---------
https://www.analyticsvidhya.com/blog/2020/07/10-techniques-to-deal-with-class-imbalance-in-machine-learning/
https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.over_sampling.RandomOverSampler.html#imblearn.over_sampling.RandomOverSampler

"""

import pandas as pd
import pickle
import numpy as np

from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler, NearMiss

from csgo_cheater_detection.config.config import data_path, seed, random_state

# load data
df = pd.read_csv(f'{data_path}\\csgo_cheater_data_8_30_20_full.csv')

# Split X and y
y = df.pop('label')
X = df

# set seeds
np.random.seed(seed)

# Random Over-Sampling
ros = RandomOverSampler(random_state=random_state)
X_temp, y_temp = ros.fit_resample(X, y)

X_temp['label'] = y_temp
print(X_temp.columns)

# parameters
sampling_methods = {
	'random_over': RandomOverSampler(),
	'SMOTE': SMOTE(),
	'random_under': RandomUnderSampler(),
	'near_miss': NearMiss()
}

# loop
for name, method in sampling_methods.items():
	# initiate the method
	engine = method

	# resample
	X_temp, y_temp = engine.fit_resample(X, y)
	X_temp['label'] = y_temp

	# save
	X_temp.to_csv(f'{data_path}\\csgo_cheater_data_{name}.csv')
