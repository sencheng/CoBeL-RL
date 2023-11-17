# basic imports
import numpy as np
# tensorflow
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import MeanSquaredError, Huber
from tensorflow.keras.optimizers.legacy import Adam
# CoBeL-RL
from cobel.networks.network_tensorflow import SequentialKerasNetwork


def build_model(input_size: int, output_size: int):
    model = Sequential()
    model.add(Dense(input_dim=input_size, units=64, activation='tanh', name='layer_dense_1'))
    model.add(Dense(units=64, activation='tanh', name='layer_dense_2'))
    model.add(Dense(units=output_size, activation='sigmoid', name='layer_output'))
    model.compile('adam', 'mse')
    
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
    model = SequentialKerasNetwork(network)
    # test default parameters
    assert type(model.model.loss) is MeanSquaredError or model.model.loss == 'mse', 'Default loss wasn\'t of type MSELoss.'
    assert type(model.model.optimizer) is Adam, 'Default optimizer wasn\'t of type Adam.'
    assert len(model.get_weights()) == 6, 'The number of weights doesn\'t match the expected value of 6.'
    
    # init model with parameters
    model = SequentialKerasNetwork(network, optimizer, loss, optimizer_params, loss_params)
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
    