"""
Batch info
----------
batch_date: August 30, 2020
number of parts: 5
number of steam IDs scraped: 50000
"""

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.metrics import Precision, Recall

from csgo_cheater_detection.config.config import data_path

# parameters
data_path = data_path

# load data
df = pd.read_csv(f'{data_path}\\csgo_cheater_data_8_30_20_full.csv')

# split X and y
X = df.drop('label', axis=1)
y_normal = df.label
y = to_categorical(y_normal)

# scale X
scaler = StandardScaler()
X = scaler.fit_transform(X)

# train test split
X_train, X_test, y_train, y_test = train_test_split(
	X, y_normal, test_size=0.2, random_state=41
)

# fit logistic regression
clf = LogisticRegression(random_state=25)
clf.fit(X_train, y_train)

# classification report
y_pred = clf.predict(X_test)
print(classification_report(
	y_test, y_pred
))

# CNN
# train test split
X_train, X_test, y_train, y_test = train_test_split(
	X, y, test_size=0.2, random_state=41
)

# configs for neural network
layer_1_nodes = 189
layer_2_nodes = 350
layer_3_nodes = 350
layer_4_nodes = 350
layer_5_nodes = 95
output_nodes = 2
input_dim = X.shape[1]

# Build network
model = Sequential()
model.add(Dense(layer_1_nodes, activation='relu', input_dim=input_dim))
model.add(Dense(layer_2_nodes, activation='relu'))
model.add(Dense(layer_3_nodes, activation='relu'))
model.add(Dense(layer_4_nodes, activation='relu'))
model.add(Dense(layer_5_nodes, activation='relu'))
model.add(Dense(output_nodes, activation='sigmoid'))

# Compile model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=[Precision(), Recall()])

# Fit model
model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))
