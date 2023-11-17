# basic imports
import numpy as np
# tensorflow
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers.legacy import Adam


# build a simple model
model = Sequential()
model.add(Dense(input_dim=2, units=64, activation='tanh', name='layer_dense_1'))
model.add(Dense(units=64, activation='tanh', name='layer_dense_2'))
model.add(Dense(units=1, activation='sigmoid', name='layer_output'))
model.compile(Adam(), MeanSquaredError())

# prepare simple XOR data set
data = np.random.rand(1000, 2)
labels = ((data[:, 0] > 0.5) * (data[:, 1] > 0.5)).astype(float) + ((data[:, 0] < 0.5) * (data[:, 1] < 0.5).astype(float))

# train the model
preds_before = model.predict_on_batch(data).flatten()
error_before = np.mean((preds_before - labels) ** 2)
for epoch in range(100):
    for batch in range(20):
        start, end = batch * 50, (batch + 1) * 50
        model.train_on_batch(data[start:end], labels[start:end])
preds_after = model.predict_on_batch(data).flatten()
error_after = np.mean((preds_after - labels) ** 2)
assert error_after < error_before, 'Model didn\'t train properly.'
