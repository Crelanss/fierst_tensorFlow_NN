import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tf.random.set_seed(1)


def normalize_input(input_tensor):
    normalized_input_tensor = np.zeros(np.shape(input_tensor))
    for i in range(np.shape(input_tensor)[0]):
        for j in range(np.shape(input_tensor)[1]):
            normalized_input_tensor[i][j] = (input_tensor[i, j] - tf.reduce_min(input_tensor, axis=0)[j]) / (tf.reduce_max(input_tensor, axis=0)[j] - tf.reduce_min(input_tensor, axis=0)[j])

    return normalized_input_tensor


def normalize_output(input_tensor):
    normalized_output_tensor = np.zeros(np.shape(input_tensor))
    for i in range(np.shape(input_tensor)[0]):
            normalized_output_tensor[i] = (input_tensor[i] - tf.reduce_min(input_tensor, axis=0)) / (tf.reduce_max(input_tensor, axis=0) - tf.reduce_min(input_tensor, axis=0))

    return normalized_output_tensor


def denormalize_output(output_normalized_tensor, output_tensor):
    denormalized_output_tensor = np.zeros(np.shape(output_normalized_tensor))
    for i in range(np.shape(output_normalized_tensor)[0]):
        denormalized_output_tensor[i] = tf.reduce_min(output_tensor, axis=0) + output_normalized_tensor[i] * (tf.reduce_max(output_tensor, axis=0) - tf.reduce_min(output_tensor, axis=0))

    return denormalized_output_tensor


house_data_file_path = './inputs/melb_data.csv'
house_data = pd.read_csv(house_data_file_path)
house_data = house_data.dropna(axis=0)

features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude', 'Price']
X = house_data[features].head(300)
y = house_data.Price.head(300)
X = np.array(X)
Y = np.array(y)

x_test = np.array(house_data[features].tail(10))
y_test = np.array(house_data.Price.tail(10))

inputs = keras.Input(shape=(np.shape(X)[1]))

dense = layers.Dense(128, activation="relu", name="first_hidden_layer")(inputs)
x = layers.Dense(64, activation="relu", name="second_hidden_layer")(dense)
outputs = layers.Dense(1, activation="tanh")(x)

model = keras.Model(inputs=inputs, outputs=outputs, name="melbourn_model")

model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.MeanSquaredError(),
    # metrics=[keras.metrics.SparseCategoricalAccuracy()]
)

X_normalized = normalize_input(X)
Y_normalized = normalize_output(Y)
x_test_normalized = normalize_input(x_test)
y_test_normalized = normalize_output(y_test)

history = model.fit(
    X_normalized,
    Y_normalized,
    batch_size=100,
    epochs=100,
    validation_split=0.2
)
print('predicting for 10 last houses')
print(y.tail(10))
print(denormalize_output(model.predict(x_test_normalized), Y))