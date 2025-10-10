# basic imports
import numpy as np
# Flax
from flax.nnx import Module, Linear, Rngs, sigmoid, tanh
# CoBeL-RL
from cobel.network import FlaxNetwork


class Model(Module):
    def __init__(self, input_size: int | tuple, output_size: int):
        super().__init__()
        rngs = Rngs(0)
        input_units: int
        if type(input_size) is int:
            input_units = input_size
        else:
            input_units = int(np.prod(input_size))
        self.layer_dense_1 = Linear(input_units, 64, rngs=rngs)
        self.layer_dense_2 = Linear(64, 64, rngs=rngs)
        self.layer_output = Linear(64, output_size, rngs=rngs)

    def __call__(self, layer_input):
        x = self.layer_dense_1(layer_input)
        x = tanh(x)
        x = self.layer_dense_2(x)
        x = tanh(x)
        x = self.layer_output(x)
        x = sigmoid(x)

        return x


if __name__ == '__main__':
    # prepare a simple data set
    rng = np.random.default_rng()
    data = rng.random((1000, 2))
    threshold = 0.5
    labels = ((data[:, 0] > threshold) * (data[:, 1] > threshold)).astype(float)
    labels += (data[:, 0] < threshold) * (data[:, 1] < threshold).astype(float)
    labels = labels.reshape((1000, 1))
    # init network
    network = Model(2, 1)

    # init model with default parameters
    model = FlaxNetwork(network)

    # test training
    weights_before = model.get_weights()
    preds_before = model.predict_on_batch(data).flatten()
    error_before = np.mean((preds_before - labels.flatten()) ** 2)
    for _ in range(100):
        for batch in range(20):
            start, end = batch * 50, (batch + 1) * 50
            model.train_on_batch(data[start:end], labels[start:end])
    preds_after = model.predict_on_batch(data).flatten()
    error_after = np.mean((preds_after - labels.flatten()) ** 2)
    print(error_before, error_after)
    assert error_after < error_before, "Model didn't train properly."
