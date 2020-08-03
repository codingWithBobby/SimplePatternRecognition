from tensorflow import keras
import numpy as np
import math

model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

model.compile(optimizer='sgd', loss='mean_squared_error')

X = np.array([-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17], dtype=int)
Y = np.array([-3, -1, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33], dtype=int)

# y = 2x -1

model.fit(X, Y, epochs=500)

prediction = model.predict([18])
print('original prediciton: ', prediction)

rounded_prediction = math.ceil(prediction)
print('rounded prediction: ', rounded_prediction)