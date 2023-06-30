import os
import pandas as pd
import glob
from os.path import basename
import numpy as np
from shapely.geometry import Polygon
import xml.etree.ElementTree as ET
import shutil
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pickle
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import xml.etree.ElementTree as ET

# Define the input and output paths obtained from gen_trapez_io.py

input = pd.read_csv('./input.csv')
output = pd.read_csv('./output.csv')

weights_file_name = "trapezium.pickle"
hidden_layer = (512, 256)
learning_rate_init = 1e-3


input = input.drop("Unnamed: 0", axis=1)
output = output.drop("Unnamed: 0", axis=1)

output['center_x'] = (output['0'] + output['2'] + output['4'] +output['6'])/4
output['center_y'] = (output['1'] + output['3'] + output['5'] +output['7'])/4
output['left_top_offset'] = output['3'] - (input['1'] + input['3']/2)
output['right_top_offset'] = output['5'] - (input['1'] + input['3']/2)
output['height'] = (output['4']+output['6'])/2-(output['0']+output['2'])/2#'width'
output['left_bottom_offset'] = output['1'] - (input['1'] - input['3']/2)
output['right_bottom_offset'] = output['7'] - (input['1'] - input['3']/2)

y = output.to_numpy()
x = input.to_numpy()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

regr = MLPRegressor(hidden_layer_sizes=hidden_layer,activation='tanh', max_iter=10000, learning_rate='adaptive', learning_rate_init=learning_rate_init).fit(x_train, y_train)

pickle.dump(regr, open(weights_file_name, 'wb'))

y_pred = regr.predict(x_test)

print('Mean squared error: %.2f'% mean_squared_error(y_test, y_pred))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'% r2_score(y_test, y_pred))