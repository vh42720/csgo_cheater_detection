"""
Batch info
----------
batch_date: August 30, 2020
number of parts: 5
number of steam IDs scraped: 50000

Goals
-----
Built and optimized 3 machine learning models
for cheater detections. Again, the result here is
for model evaluation and is not for used to place
a VAC ban. In application, the model will be adjusted
to return probability of someone is a cheater or not.

The question of how high of a probability should a
player is placed on the watch list is complicated.
It will be discussed more on the report.

Metrics
-------
Since the data is imbalanced (with 85%+ non cheaters),
model performance depends on 3 metrics.
	- Precision = TP / (TP + FP)
	- Recall = TP / (TP + FN)
	- Balanced accuracy = (TPR + TNR) / 2
	- ROC/AUC score = area under the curve

Structures
----------
The script is broken down into 4 parts
	1. Preparing data
	2. Logistic Classifier - baseline
	3. Random forest
	4. Neural Network
For reproducibility, random_state = 25, seed = 25
Label clarification:
	0: VAC ban is False / Non cheater
	1: VAC ban is True / Cheater

Reference
---------
https://en.wikipedia.org/wiki/Precision_and_recall
https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc
https://en.wikipedia.org/wiki/Logistic_regression
https://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm#intro
https://en.wikipedia.org/wiki/Neural_network#:~:text=A%20neural%20network%20(NN)%2C,a%20connectionistic%20approach%20to%20computation.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import classification_report, roc_auc_score, \
	precision_recall_curve, plot_precision_recall_curve
from sklearn.linear_model import LogisticRegression

from csgo_cheater_detection.config.config import *
from csgo_cheater_detection.utils.functions import split_X_y, \
	best_random_forest_clf, make_neural_network, plot_roc

"""
Prepping X and y
----------------
Split data set with 80% training and 20% testing.
Split training data into 80% training and 20% validation.	
X_train is standardize using normal scaler from sklearn. This will
rescale X to have a mean of 0 and standard deviation of 1.

It is a good practice to fit the scaler on the training data 
and then use it to transform the testing data. This would avoid 
any data leakage during the model testing process. 
The scaling of target values is generally not required.

Three partition of data: train, validation and test.
The test set is unused during training phase and is only
used at the end to evaluate model.

Reference
---------
https://www.analyticsvidhya.com/blog/2020/04/feature-scaling-machine-learning-normalization-standardization/
split_X_y from utils.functions.split_X_y
"""

# load data
sampling_df, sampling_score = sampling_dict.copy(), result_dict.copy()
for name in sampling_df.keys():
	sampling_df[name] = pd.read_csv(
		f'{data_path}\\csgo_cheater_data_{name}.csv',
		index_col=[0]
	)

# set seed = 25
np.random.seed(seed)

for name, df in sampling_df.items():
	# number of positive and negative
	neg, pos = df.label.value_counts()
	total = neg + pos

	# Split X and y
	X_train, X_val, X_test, \
		y_train, y_val, y_test = split_X_y(
			df=df,
			label_col='label',
			test_size=0.2,
			random_state=random_state)

	"""
	Logistic Regression
	-------------------
	The fast and popular logistic regression provide
	a baseline performance metrics without optimizations.
	"""

	# Logistic regression with 'lbgfs' solver
	clf = LogisticRegression(
		random_state=41,
		max_iter=10000
	)
	clf.fit(X_train, y_train)

	# Logistic regression performance
	y_pred = clf.predict(X_test)
	sampling_score[name]['logistic']['report'] = classification_report(y_test, y_pred)
	sampling_score[name]['logistic']['roc_auc_score'] = roc_auc_score(y_test, y_pred)

	# Save the plot
	disp = plot_precision_recall_curve(clf, X_test, y_test)
	disp.ax_.set_title(f'{name} sampling logistic precision-recall curve')
	plt.savefig(f'{image_path}\\{name}_logistic_prc.png')
	plt.clf()

	"""
	Random Forest
	-------------
	Tree based algorithm should perform well with
	an unbalanced data set. Implementation through
	GridSearchCV to tune hyper-parameters over
	32 settings (see configs.py).
	Refit with the best parameters and save
	model performance on test data.
	
	Reference
	---------
	https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
	https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
	"""

	# define the parameters grid for GridSearchCV
	param_grid = rf_grid

	# Initiate random forest
	rf = best_random_forest_clf(rf_grid, X_train, y_train)

	# Random forest performance
	y_pred = rf.predict(X_test)
	sampling_score[name]['random_forest']['report'] = classification_report(y_test, y_pred)
	sampling_score[name]['random_forest']['roc_auc_score'] = roc_auc_score(y_test, y_pred)

	# Save the plot
	disp = plot_precision_recall_curve(rf, X_test, y_test)
	disp.ax_.set_title(f'{name} sampling random forest precision-recall curve')
	plt.savefig(f'{image_path}\\{name}_random_forest_prc.png')
	plt.clf()

	"""
	Neural Network
	--------------
	Surprisingly, the training time for neural network is incredibly
	fast with the gpu support. Only a simple dense structure is 
	implemented.
	
	Initial bias
	------------
	Since the dataset is known to be imbalanced, setting layers' bias
	can help greatly. The formula:
		initial bias = ln(positive / negative)
	
	Class weights
	-------------
	Weighting the importance of each label also helps when it comes 
	to imbalanced dataset: lesser label is given a higher weights
	in the network than the popular label. The formula:
		weight_for_0 = (1 / neg)*(total)/2.0
		weight_for_1 = (1 / pos)*(total)/2.0
	
	Reference
	---------
	https://www.tensorflow.org/tutorials/structured_data/imbalanced_data
	"""

	input_dim = X_train.shape[-1]

	# Calculate the initial bias for output node
	initial_bias = np.log([pos / neg])

	# Create a weight for each class.
	# Scaling by total/2 helps keep the loss to a similar magnitude.
	weight_for_0 = (1 / neg) * total / 2.0
	weight_for_1 = (1 / pos) * total / 2.0
	class_weight = {0: weight_for_0, 1: weight_for_1}

	# Fit neural network
	model = make_neural_network(input_dim=X_train.shape[-1], output_bias=initial_bias)
	model.fit(
		X_train,
		y_train,
		epochs=5,
		validation_data=(X_val, y_val),
		class_weight=class_weight
	)

	# prediction
	y_pred = model.predict(X_test)
	y_pred_label = y_pred >= 0.5

	# Save reports for comparisons
	sampling_score[name]['neural_network']['report'] = classification_report(y_test, y_pred_label)
	sampling_score[name]['neural_network']['roc_auc_score'] = roc_auc_score(y_test, y_pred_label)

	# save plot
	disp = plot_roc(f'{name} sampling neural network ROC curve', y_test, y_pred)
	plt.savefig(f'{image_path}\\{name}_neural_network_prc.png')
	plt.clf()
