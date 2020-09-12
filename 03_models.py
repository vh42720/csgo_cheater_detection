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

Structures
----------
The script is broken down into 4 parts
	1. Preparing data
	2. Logistic Classifier - baseline
	3. Random forest
	4. Neural Network
For reproducibility, random_state = 25
Label clarification:
	0: VAC ban is False / Non cheater
	1: VAC ban is True / Cheater

Reference
---------
https://en.wikipedia.org/wiki/Precision_and_recall
https://en.wikipedia.org/wiki/Logistic_regression
https://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm#intro
https://en.wikipedia.org/wiki/Neural_network#:~:text=A%20neural%20network%20(NN)%2C,a%20connectionistic%20approach%20to%20computation.
"""

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, balanced_accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

import tensorflow as tf
from tensorflow import keras

from csgo_cheater_detection.config.config import data_path

"""
Prepping X and y
----------------
Split data set with 80% training and 20% testing.	
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
"""

# load data
df = pd.read_csv(f'{data_path}\\csgo_cheater_data_8_30_20_full.csv')

# number of positive and negative
neg, pos = df.label.value_counts()
total = neg + pos

# Split and shuffle datasets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=41)
train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=41)

# Form X and y for each dataset
y_train = train_df.pop('label')
y_val = val_df.pop('label')
y_test = test_df.pop('label')

X_train, X_val, X_test = train_df, val_df, test_df

# standardize training data and transform test data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

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
print('Logistic Classifier')
print(classification_report(y_test, y_pred))
print('Balance accuracy score:', balanced_accuracy_score(y_test, y_pred))

"""
Random Forest
-------------
Tree based algorithm should perform well with
an unbalanced data set. Implementation through
GridSearchCV to tune hyper-parameters over
1 * 2 * 2 * 3 * 3 * 3 = 108 settings.
Refit with the best parameters and print out
model performance on test data.

Reference
---------
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
"""

# define the parameters grid for GridSearchCV
param_grid = {
	'bootstrap': [True],
	'max_depth': [80, 90],
	'max_features': [2, 3],
	'min_samples_leaf': [3, 4, 5],
	'min_samples_split': [8, 10, 12],
	'n_estimators': [100, 300, 500]
}

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

# Random forest performance
y_pred = rf.predict(X_test)
print('Random Forest')
print(classification_report(y_test, y_pred))
print('Balance accuracy score:', balanced_accuracy_score(y_test, y_pred))

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

Reference
---------
https://www.tensorflow.org/tutorials/structured_data/imbalanced_data
"""

# Define metrics and model configs
METRICS = [
# 	keras.metrics.TruePositives(name='tp'),
# 	keras.metrics.FalsePositives(name='fp'),
# 	keras.metrics.TrueNegatives(name='tn'),
# 	keras.metrics.FalseNegatives(name='fn'),
	keras.metrics.BinaryAccuracy(name='accuracy'),
	keras.metrics.Precision(name='precision'),
	keras.metrics.Recall(name='recall'),
	keras.metrics.AUC(name='auc'),
]

# configs for neural network
layer_1_nodes = 350
layer_2_nodes = 350
layer_3_nodes = 350
layer_4_nodes = 350
layer_5_nodes = 95
output_nodes = 1
input_dim = X_train.shape[-1]

# Calculate the initial bias for output node
initial_bias = np.log([pos/neg])

# Create a weight for each class.
# Scaling by total/2 helps keep the loss to a similar magnitude.
weight_for_0 = (1 / neg) * total / 2.0
weight_for_1 = (1 / pos) * total / 2.0
class_weight = {0: weight_for_0, 1: weight_for_1}


def make_model(metrics=None, output_bias=None):
	"""
	Define a model with 5 dense layers, a dropout layer to
	reduce overfitting and a sigmoid output node that returns
	probability of a player being a cheater.

	:param metrics: list of metrics.
	:param output_bias: initial bias.
	:return: neural network ready for training.
	"""
	if metrics is None:
		metrics = METRICS
	if output_bias is not None:
		output_bias = tf.keras.initializers.Constant(output_bias)
	model = keras.Sequential([
		keras.layers.Dense(layer_1_nodes, activation='relu', input_dim=input_dim),
		keras.layers.Dense(layer_2_nodes, activation='relu'),
		keras.layers.Dense(layer_3_nodes, activation='relu'),
		keras.layers.Dense(layer_4_nodes, activation='relu'),
		keras.layers.Dense(layer_5_nodes, activation='relu'),
		keras.layers.Dropout(0.5),
		keras.layers.Dense(1, activation='sigmoid', bias_initializer=output_bias),
	])

	model.compile(
		optimizer=keras.optimizers.Adam(lr=1e-3),
		loss=keras.losses.BinaryCrossentropy(),
		metrics=METRICS)

	return model


# Fit model
model = make_model(output_bias=initial_bias)
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

# Classification reports
print('Neural Network')
print(classification_report(y_test, y_pred_label))
print('Balance accuracy score:', balanced_accuracy_score(y_test, y_pred_label))

# Evaluation of neural network
evals = model.evaluate(X_test, y_test, verbose=0)
for name, value in zip(model.metrics_names, evals):
	print(name, ': ', value)
