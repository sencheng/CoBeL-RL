# basic imports
import numpy as np
# tensorflow
import tensorflow as tf  # type: ignore
from tensorflow.keras.models import Sequential, Model  # type: ignore
from tensorflow.keras.layers import Dense, concatenate  # type: ignore
from tensorflow.keras.losses import MeanSquaredError, Huber  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore
# CoBeL-RL
from cobel.network import KerasNetwork


def build_model(input_size: int, output_size: int) -> Sequential:
    model = Sequential()
    model.add(tf.keras.Input(shape=(input_size,)))
    model.add(Dense(units=64, activation='tanh', name='layer_dense_1'))
    model.add(Dense(units=64, activation='tanh', name='layer_dense_2'))
    model.add(Dense(units=output_size, activation='sigmoid', name='layer_output'))
    model.compile('adam', 'mse')

    return model


def build_model_multi(input_shapes: tuple, output_sizes: tuple) -> Model:
    layer_input_1 = tf.keras.Input(shape=(input_shapes[0],), name='layer_input_1')
    layer_input_2 = tf.keras.Input(shape=(input_shapes[1],), name='layer_input_2')
    stream_in_1_1 = Dense(units=64, activation='tanh', name='stream_in_1_1')(
        layer_input_1
    )
    stream_in_1_2 = Dense(units=32, activation='tanh', name='stream_in_1_2')(
        stream_in_1_1
    )
    stream_in_2_1 = Dense(units=64, activation='tanh', name='stream_in_2_1')(
        layer_input_2
    )
    stream_in_2_2 = Dense(units=32, activation='tanh', name='stream_in_2_2')(
        stream_in_2_1
    )
    concat = concatenate([stream_in_1_2, stream_in_2_2], name='layer_concatenate')
    hidden = Dense(units=32, activation='tanh', name='layer_hidden')(concat)
    stream_out_1_1 = Dense(units=32, activation='tanh', name='stream_out_1_1')(hidden)
    stream_out_1_2 = Dense(
        units=output_sizes[0], activation='sigmoid', name='stream_out_1_2'
    )(stream_out_1_1)
    stream_out_2_1 = Dense(units=32, activation='tanh', name='stream_out_2_1')(hidden)
    stream_out_2_2 = Dense(
        units=output_sizes[1], activation='sigmoid', name='stream_out_2_2'
    )(stream_out_2_1)
    model = Model(
        inputs=[layer_input_1, layer_input_2],
        outputs=[stream_out_1_2, stream_out_2_2],
        name='multi_model',
    )
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy', 'accuracy'])

    return model


if __name__ == '__main__':
    # prepare a simple data set
    data = np.random.default_rng().random((1000, 2))
    labels = ((data[:, 0] > 0.5) * (data[:, 1] > 0.5)).astype(float) + (
        (data[:, 0] < 0.5) * (data[:, 1] < 0.5).astype(float)
    )
    # init network
    network = build_model(2, 1)
    # prepare training hyper-parameters
    loss = Huber()
    loss_params = {'delta': 1.1}
    optimizer = Adam()
    optimizer_params = {'learning_rate': 0.005, 'beta_1': 0.91, 'beta_2': 0.998}

    # init model with default parameters
    model = KerasNetwork(network)
    # test default parameters
    assert type(model.model.loss) is MeanSquaredError or model.model.loss == 'mse', (
        "Default loss wasn't of type MSELoss."
    )
    assert type(model.model.optimizer) is Adam, "Default optimizer wasn't of type Adam."
    assert len(model.get_weights()) == 6, (
        "The number of weights doesn't match the expected value of 6."
    )

    # init model with parameters
    model = KerasNetwork(network)
    model.set_optimizer(optimizer, optimizer_params)
    model.set_loss(loss, loss_params)
    # test model parameters
    assert type(model.model.loss) is Huber, "Model loss wasn't of type HuberLoss."
    assert type(model.model.optimizer) is Adam, "Model optimizer wasn't of type Adam."
    config = model.model.optimizer.get_config()
    assert (
        abs(config['learning_rate'] - 0.005) < 10**-8
        and config['beta_1'] == 0.91
        and config['beta_2'] == 0.998
    ), (
        "Model optimizer's parameters don't match the expected values"
        " (lr==0.005, betas==(0.91, 0.998))."
    )

    # test layer activity retrieval
    # with layer indices
    for i, size in enumerate([64, 64, 1]):
        layer_shape = model.get_layer_activity(data[:10], i).shape
        assert layer_shape == (size, 10), (
            "Shape of activity at layer %d didn't match the expected shape of (%d, 10)."
            % (i, size)
        )
    # with layer names
    for layer, size in {
        'layer_dense_1': 64,
        'layer_dense_2': 64,
        'layer_output': 1,
    }.items():
        layer_shape = model.get_layer_activity(data[:10], layer).shape
        assert layer_shape == (size, 10), (
            "Shape of activity at layer %s didn't match the expected shape of (%d, 10)."
            % (layer, size)
        )

    # test training
    weights_before = model.get_weights()
    preds_before = model.predict_on_batch(data).flatten()  # type: ignore
    error_before = np.mean((preds_before - labels) ** 2)
    for _ in range(100):
        for batch in range(20):
            start, end = batch * 50, (batch + 1) * 50
            model.train_on_batch(data[start:end], labels[start:end])
    preds_after = model.predict_on_batch(data).flatten()  # type: ignore
    error_after = np.mean((preds_after - labels) ** 2)
    assert error_after < error_before, "Model didn't train properly."
    # test weight setting
    model.set_weights(weights_before)
    for i, weight in enumerate(model.get_weights()):
        assert np.array_equal(weights_before[i], weight), (
            'Weights not set at layer %d.' % i
        )

    # test model cloning
    model_cloned = model.clone()
    assert type(model.model.loss) is type(model_cloned.model.loss), (
        "Cloned model's loss function doesn't match."
    )
    assert type(model.model.optimizer) is type(model_cloned.model.optimizer), (
        "Cloned model's optimizer doesn't match."
    )
    config, config_cloned = (
        model.model.optimizer.get_config(),
        model_cloned.model.optimizer.get_config(),
    )
    assert (
        config['learning_rate'] == config_cloned['learning_rate']
        and config['beta_2'] == config_cloned['beta_2']
        and config['beta_2'] == config_cloned['beta_2']
    ), "Cloned model's optimizer's parameter don't match."
    assert model.model.layers != model_cloned.model.layers, (
        'Cloned model shares underlying parameters.'
    )

    # test setters
    # loss function
    model.set_loss('mse')
    assert type(model.model.loss) == MeanSquaredError, (
        "Loss function wasn't set when using string identifier."
    )
    model.set_loss('huber', loss_params)
    model.set_loss('mse')
    model.set_loss(loss, loss_params)
    assert type(model.model.loss) is Huber, (
        "Loss function wasn't set when using loss class."
    )
    # optimizer
    model.set_optimizer('adam', {'learning_rate': 0.1})
    assert type(model.model.optimizer) is Adam, (
        "Optimizer wasn't set when using string identifier"
    )
    assert abs(model.model.optimizer.get_config()['learning_rate'] - 0.1) < 10**-8, (
        "Optimizer's lr parameter wasn't set when using string identifier."
    )
    model.set_optimizer(Adam(), {'learning_rate': 0.01})
    assert type(model.model.optimizer) is Adam, (
        "Optimizer wasn't set when using string identifier"
    )
    assert abs(model.model.optimizer.get_config()['learning_rate'] - 0.01) < 10**-8, (
        "Optimizer's lr parameter wasn't set when using string identifier."
    )

    #
    # init network
    network = build_model_multi((2, 2), (1, 1))
    # prepare training hyper-parameters
    optimizer = Adam()
    optimizer_params = {'learning_rate': 0.005, 'beta_1': 0.91, 'beta_2': 0.998}

    # init model with default parameters
    model = KerasNetwork(network)
    # test default parameters
    assert type(model.model.loss) is MeanSquaredError or model.model.loss == 'mse', (
        "Default loss wasn't of type MSELoss."
    )
    assert type(model.model.optimizer) is Adam, "Default optimizer wasn't of type Adam."
    assert len(model.get_weights()) == 18, (
        "The number of weights doesn't match the expected value of 6."
    )

    # init model with parameters
    model = KerasNetwork(network)
    model.set_optimizer(optimizer, optimizer_params)
    # test parameters
    assert type(model.model.optimizer) is Adam, "Default optimizer wasn't of type Adam."
    config = model.model.optimizer.get_config()
    assert (
        abs(config['learning_rate'] - 0.005) < 10**-8
        and config['beta_1'] == 0.91
        and config['beta_2'] == 0.998
    ), (
        "Model optimizer's parameters don't match the expected values"
        " (lr==0.005, betas==(0.91, 0.998))."
    )
    assert len(model.get_weights()) == 18, (
        "The number of weights doesn't match the expected value of 6."
    )

    # test layer activity retrieval
    # with layer indices
    for i, size in enumerate([64, 64, 32, 32, 64, 32, 32, 32, 1, 1]):
        layer_shape = model.get_layer_activity([data[:10], data[:10]], i + 2).shape
        assert layer_shape == (size, 10), (
            "Shape of activity at layer %d didn't match the expected shape of (%d, 10)."
            % (i, size)
        )
    # with layer names
    for layer, size in {
        'stream_in_1_1': 64,
        'stream_in_1_2': 32,
        'stream_in_2_1': 64,
        'stream_in_2_2': 32,
        'layer_hidden': 32,
        'stream_out_1_1': 32,
        'stream_out_1_2': 1,
        'stream_out_2_1': 32,
        'stream_out_2_2': 1,
    }.items():
        layer_shape = model.get_layer_activity([data[:10], data[:10]], layer).shape
        assert layer_shape == (size, 10), (
            "Shape of activity at layer %s didn't match the expected shape of (%d, 10)."
            % (layer, size)
        )

    # test training
    weights_before = model.get_weights()
    preds_before = model.predict_on_batch([data, data])
    error_before = np.mean((preds_before[0].flatten() - labels) ** 2) + np.mean( # type: ignore
        (preds_before[1].flatten() - labels) ** 2 # type: ignore
    )
    for _ in range(100):
        for batch in range(20):
            start, end = batch * 50, (batch + 1) * 50
            model.train_on_batch(
                [data[start:end], data[start:end]],
                [labels[start:end], labels[start:end]],
            )
    preds_after = model.predict_on_batch([data, data])
    error_after = np.mean((preds_after[0].flatten() - labels) ** 2) + np.mean( # type: ignore
        (preds_after[1].flatten() - labels) ** 2 # type: ignore
    )
    assert error_after < error_before, "Model didn't train properly."
    # test weight setting
    model.set_weights(weights_before)
    for i, weight in enumerate(model.get_weights()):
        assert np.array_equal(weights_before[i], weight), (
            'Weights not set at layer %d.' % i
        )

    # test model cloning
    model_cloned = model.clone()
    assert type(model.model.optimizer) is type(model_cloned.model.optimizer), (
        "Cloned model's optimizer doesn't match."
    )
    config, config_cloned = (
        model.model.optimizer.get_config(),
        model_cloned.model.optimizer.get_config(),
    )
    assert (
        config['learning_rate'] == config_cloned['learning_rate']
        and config['beta_2'] == config_cloned['beta_2']
        and config['beta_2'] == config_cloned['beta_2']
    ), "Cloned model's optimizer's parameter don't match."
    assert model.model.layers != model_cloned.model.layers, (
        'Cloned model shares underlying parameters.'
    )
