# basic imports
import numpy as np
# tensorflow
import tensorflow as tf
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, concatenate
from tensorflow.keras.losses import MeanSquaredError, Huber
from tensorflow.keras.optimizers.legacy import Adam
# CoBeL-RL
from cobel.networks.network_tensorflow import FunctionalKerasNetwork


def build_model(input_size: int, output_size: int):
    model = Sequential()
    model.add(Dense(input_dim=input_size, units=64, activation='tanh', name='layer_dense_1'))
    model.add(Dense(units=64, activation='tanh', name='layer_dense_2'))
    model.add(Dense(units=output_size, activation='sigmoid', name='layer_output'))
    model.compile('adam', 'mse')
    
    return model

def build_model_multi(input_shapes: tuple, output_sizes: tuple):
    layer_input_1 = tf.keras.Input(shape=input_shapes[0], name='layer_input_1')
    layer_input_2 = tf.keras.Input(shape=input_shapes[1], name='layer_input_2')
    stream_in_1_1 = Dense(units=64, activation='tanh', name='stream_in_1_1')(layer_input_1)
    stream_in_1_2 = Dense(units=32, activation='tanh', name='stream_in_1_2')(stream_in_1_1)
    stream_in_2_1 = Dense(units=64, activation='tanh', name='stream_in_2_1')(layer_input_2)
    stream_in_2_2 = Dense(units=32, activation='tanh', name='stream_in_2_2')(stream_in_2_1)
    concat = concatenate([stream_in_1_2, stream_in_2_2], name='layer_concatenate')
    hidden = Dense(units=32, activation='tanh', name='layer_hidden')(concat)
    stream_out_1_1 = Dense(units=32, activation='tanh', name='stream_out_1_1')(hidden)
    stream_out_1_2 = Dense(units=output_sizes[0], activation='sigmoid', name='stream_out_1_2')(stream_out_1_1)
    stream_out_2_1 = Dense(units=32, activation='tanh', name='stream_out_2_1')(hidden)
    stream_out_2_2 = Dense(units=output_sizes[1], activation='sigmoid', name='stream_out_2_2')(stream_out_2_1)
    model = Model(inputs=[layer_input_1, layer_input_2], outputs=[stream_out_1_2, stream_out_2_2], name='multi_model')
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    
    return model


if __name__ == '__main__':
    # prepare a simple data set
    data = np.random.rand(1000, 2)
    labels = ((data[:, 0] > 0.5) * (data[:, 1] > 0.5)).astype(float) + ((data[:, 0] < 0.5) * (data[:, 1] < 0.5).astype(float))
    # init network
    network = build_model(2, 1)
    # prepare training hyper-parameters
    loss = Huber()
    loss_params = {'delta': 1.1}
    optimizer = Adam()
    optimizer_params = {'learning_rate': 0.005, 'beta_1': 0.91, 'beta_2': 0.998}
    
    # init model with default parameters
    model = FunctionalKerasNetwork(network)
    # test default parameters
    assert type(model.model.loss) is MeanSquaredError or model.model.loss == 'mse', 'Default loss wasn\'t of type MSELoss.'
    assert type(model.model.optimizer) is Adam, 'Default optimizer wasn\'t of type Adam.'
    assert len(model.get_weights()) == 6, 'The number of weights doesn\'t match the expected value of 6.'
    
    # init model with parameters
    model = FunctionalKerasNetwork(network, optimizer, loss, optimizer_params, loss_params)
    # test model parameters
    assert type(model.model.loss) is Huber, 'Model loss wasn\'t of type HuberLoss.'
    assert model.model.loss.get_config()['delta'] == 1.1, 'Huber loss\' delta parameter doesn\'t match expected value of 1.1.'
    assert type(model.model.optimizer) is Adam, 'Model optimizer wasn\'t of type Adam.'
    config = model.model.optimizer.get_config()
    assert config['learning_rate'] == 0.005 and config['beta_1'] == 0.91 and config['beta_2'] == 0.998, 'Model optimizer\'s parameters don\'t match the expected values (lr==0.005, betas==(0.91, 0.998)).'
    
    # test layer activity retrieval
    # with layer indices
    for i, size in enumerate([64, 64, 1]):
        layer_shape = model.get_layer_activity(data[:10], i).shape
        assert layer_shape == (size, 10), 'Shape of activity at layer %d didn\'t match the expected shape of (%d, 10).' % (i, size)
    # with layer names
    for layer, size in {'layer_dense_1': 64, 'layer_dense_2': 64, 'layer_output': 1}.items():
        layer_shape = model.get_layer_activity(data[:10], layer).shape
        assert layer_shape == (size, 10), 'Shape of activity at layer %s didn\'t match the expected shape of (%d, 10).' % (layer, size)
        
    # test training
    weights_before = model.get_weights()
    preds_before = model.predict_on_batch(data).flatten()
    error_before = np.mean((preds_before - labels) ** 2)
    for epoch in range(100):
        for batch in range(20):
            start, end = batch * 50, (batch + 1) * 50
            model.train_on_batch(data[start:end], labels[start:end])
    preds_after = model.predict_on_batch(data).flatten()
    error_after = np.mean((preds_after - labels) ** 2)
    assert error_after < error_before, 'Model didn\'t train properly.'
    # test weight setting
    model.set_weights(weights_before)
    for i, weight in enumerate(model.get_weights()):
        assert np.array_equal(weights_before[i], weight), 'Weights not set at layer %d.' % i
    
    # test model cloning
    model_cloned = model.clone_model()
    assert type(model.model.loss) == type(model_cloned.model.loss), 'Cloned model\'s loss function doesn\'t match.'
    assert model.model.loss.get_config()['delta'] == model_cloned.model.loss.get_config()['delta'], 'Cloned model\'s loss function\'s delta parameter doesn\'t match.'
    assert type(model.model.optimizer) == type(model_cloned.model.optimizer), 'Cloned model\'s optimizer doesn\'t match.'
    config, config_cloned = model.model.optimizer.get_config(), model_cloned.model.optimizer.get_config()
    assert config['learning_rate'] == config_cloned['learning_rate'] and config['beta_2'] == config_cloned['beta_2'] and config['beta_2'] == config_cloned['beta_2'], 'Cloned model\'s optimizer\'s parameter don\'t match.'
    assert model.model.layers != model_cloned.model.layers, 'Cloned model shares underlying parameters.'
    
    # test setters
    # loss function
    model.set_loss('mse')
    assert type(model.model.loss) == MeanSquaredError, 'Loss function wasn\'t set when using string identifier.'
    model.set_loss('huber', loss_params)
    assert model.model.loss.get_config()['delta'] == 1.1, 'Loss function\'s delta parameter wasn\'t set when using string identifier.'
    model.set_loss('mse')
    model.set_loss(loss, loss_params)
    assert type(model.model.loss) is Huber, 'Loss function wasn\'t set when using loss class.'
    assert model.model.loss.get_config()['delta'] == 1.1, 'Loss function\'s delta parameter wasn\'t set when using loss class.'
    # optimizer
    model.set_optimizer('adam', {'learning_rate': 0.1})
    assert type(model.model.optimizer) is Adam, 'Optimizer wasn\'t set when using string identifier'
    assert model.model.optimizer.get_config()['learning_rate'] == 0.1, 'Optimizer\'s lr parameter wasn\'t set when using string identifier.'
    model.set_optimizer(Adam(), {'learning_rate': 0.01})
    assert type(model.model.optimizer) is Adam, 'Optimizer wasn\'t set when using string identifier'
    assert model.model.optimizer.get_config()['learning_rate'] == 0.01, 'Optimizer\'s lr parameter wasn\'t set when using string identifier.'
    
    #
    # init network
    network = build_model_multi((2, 2), (1, 1))
    # prepare training hyper-parameters
    loss = [MeanSquaredError(), Huber()]
    loss_params = [{}, {'delta': 1.1}]
    loss_weights = [0.5, 0.5]
    optimizer = Adam()
    optimizer_params = {'learning_rate': 0.005, 'beta_1': 0.91, 'beta_2': 0.998}
    
    # init model with default parameters
    model = FunctionalKerasNetwork(network)
    # test default parameters
    assert type(model.model.loss) is MeanSquaredError or model.model.loss == 'mse', 'Default loss wasn\'t of type MSELoss.'
    assert type(model.model.optimizer) is Adam, 'Default optimizer wasn\'t of type Adam.'
    assert len(model.get_weights()) == 18, 'The number of weights doesn\'t match the expected value of 6.'
    
    # init model with parameters
    model = FunctionalKerasNetwork(network, optimizer, loss, optimizer_params, loss_params, loss_weights)
    # test parameters
    assert len(model.model.loss) == 2, 'Wrong number of loss functions.'
    assert type(model.model.loss[0]) is MeanSquaredError and type(model.model.loss[1]) is Huber, 'Default loss wasn\'t of type MSELoss.'
    assert model.model.loss[1].get_config()['delta'] == 1.1, 'Huber loss\' delta parameter doesn\'t match expected value of 1.1.'
    assert type(model.model.optimizer) is Adam, 'Default optimizer wasn\'t of type Adam.'
    config = model.model.optimizer.get_config()
    assert config['learning_rate'] == 0.005 and config['beta_1'] == 0.91 and config['beta_2'] == 0.998, 'Model optimizer\'s parameters don\'t match the expected values (lr==0.005, betas==(0.91, 0.998)).'
    assert len(model.get_weights()) == 18, 'The number of weights doesn\'t match the expected value of 6.'
    
    # test layer activity retrieval
    # with layer indices
    for i, size in enumerate([64, 64, 32, 32, 64, 32, 32, 32, 1, 1]):
        layer_shape = model.get_layer_activity([data[:10], data[:10]], i + 2).shape
        assert layer_shape == (size, 10), 'Shape of activity at layer %d didn\'t match the expected shape of (%d, 10).' % (i, size)
    # with layer names
    for layer, size in {'stream_in_1_1': 64, 'stream_in_1_2': 32, 'stream_in_2_1': 64, 'stream_in_2_2': 32, 'layer_hidden': 32,
                        'stream_out_1_1': 32, 'stream_out_1_2': 1, 'stream_out_2_1': 32, 'stream_out_2_2': 1}.items():
        layer_shape = model.get_layer_activity([data[:10], data[:10]], layer).shape
        assert layer_shape == (size, 10), 'Shape of activity at layer %s didn\'t match the expected shape of (%d, 10).' % (layer, size)
    
    # test training
    weights_before = model.get_weights()
    preds_before = model.predict_on_batch([data, data])
    error_before = np.mean((preds_before[0].flatten() - labels) ** 2) + np.mean((preds_before[1].flatten() - labels) ** 2)
    for epoch in range(100):
        for batch in range(20):
            start, end = batch * 50, (batch + 1) * 50
            model.train_on_batch([data[start:end], data[start:end]], [labels[start:end], labels[start:end]])
    preds_after = model.predict_on_batch([data, data])
    error_after = np.mean((preds_after[0].flatten() - labels) ** 2) + np.mean((preds_after[1].flatten() - labels) ** 2)
    assert error_after < error_before, 'Model didn\'t train properly.'
    # test weight setting
    model.set_weights(weights_before)
    for i, weight in enumerate(model.get_weights()):
        assert np.array_equal(weights_before[i], weight), 'Weights not set at layer %d.' % i

    # test model cloning
    model_cloned = model.clone_model()
    for i, L in enumerate(model.model.loss):
        idx = L if type(L) is str else i
        assert type(L) == type(model_cloned.model.loss[idx]), 'Cloned model\'s loss function at index %s doesn\'t match.' % idx
    assert model.model.loss[1].get_config()['delta'] == model_cloned.model.loss[1].get_config()['delta'], 'Cloned model\'s loss function\'s delta parameter doesn\'t match.'
    assert type(model.model.optimizer) == type(model_cloned.model.optimizer), 'Cloned model\'s optimizer doesn\'t match.'
    config, config_cloned = model.model.optimizer.get_config(), model_cloned.model.optimizer.get_config()
    assert config['learning_rate'] == config_cloned['learning_rate'] and config['beta_2'] == config_cloned['beta_2'] and config['beta_2'] == config_cloned['beta_2'], 'Cloned model\'s optimizer\'s parameter don\'t match.'
    assert model.model.layers != model_cloned.model.layers, 'Cloned model shares underlying parameters.'
    
    # test setters
    # loss function
    model.set_loss(['huber', 'mse'])
    assert type(model.model.loss[1]) == MeanSquaredError, 'Loss function wasn\'t set when using string identifier.'
    model.set_loss(['mse', 'huber'], loss_params)
    assert model.model.loss[1].get_config()['delta'] == 1.1, 'Loss function\'s delta parameter wasn\'t set when using string identifier.'
    model.set_loss(['huber', 'mse'])
    model.set_loss(loss, loss_params)
    assert type(model.model.loss[1]) is Huber, 'Loss function wasn\'t set when using loss class.'
    assert model.model.loss[1].get_config()['delta'] == 1.1, 'Loss function\'s delta parameter wasn\'t set when using loss class.'
    # optimizer
    model.set_optimizer('adam', {'learning_rate': 0.1})
    assert type(model.model.optimizer) is Adam, 'Optimizer wasn\'t set when using string identifier'
    assert model.model.optimizer.get_config()['learning_rate'] == 0.1, 'Optimizer\'s lr parameter wasn\'t set when using string identifier.'
    model.set_optimizer(Adam(), {'learning_rate': 0.01})
    assert type(model.model.optimizer) is Adam, 'Optimizer wasn\'t set when using string identifier'
    assert model.model.optimizer.get_config()['learning_rate'] == 0.01, 'Optimizer\'s lr parameter wasn\'t set when using string identifier.'
    