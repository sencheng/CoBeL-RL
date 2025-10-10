# basic imports
import numpy as np
# torch
from torch import Tensor, reshape, set_num_threads
from torch.nn import Module, Linear, HuberLoss, MSELoss
from torch.optim import Adam
from torch.nn.functional import tanh, sigmoid
# CoBeL-RL
from cobel.network import TorchNetwork


class Model(Module):
    def __init__(self, input_size: int | tuple, output_size: int):
        super().__init__()
        input_units: int
        if type(input_size) is int:
            input_units = input_size
        else:
            input_units = int(np.prod(input_size))
        self.layer_dense_1 = Linear(in_features=input_units, out_features=64)
        self.layer_dense_2 = Linear(in_features=64, out_features=64)
        self.layer_output = Linear(in_features=64, out_features=output_size)
        self.double()

    def forward(self, layer_input: Tensor) -> Tensor:
        x = reshape(layer_input, (len(layer_input), -1))
        x = self.layer_dense_1(x)
        x = tanh(x)
        x = self.layer_dense_2(x)
        x = tanh(x)
        x = self.layer_output(x)
        x = sigmoid(x)

        return x


if __name__ == '__main__':
    set_num_threads(1)
    # prepare a simple data set
    rng = np.random.default_rng()
    data = rng.random((1000, 2))
    threshold = 0.5
    labels = ((data[:, 0] > threshold) * (data[:, 1] > threshold)).astype(float)
    labels += (data[:, 0] < threshold) * (data[:, 1] < threshold).astype(float)
    # init network
    network = Model(2, 1)
    # prepare training hyper-parameters
    loss = HuberLoss()
    loss_params = {'delta': 1.1}
    optimizer = Adam(network.parameters())
    optimizer_params = {'lr': 0.005, 'betas': (0.91, 0.998)}
    activations = {
        'layer_dense_1': tanh,
        'layer_dense_2': tanh,
        'layer_output': sigmoid,
    }

    # init model with default parameters
    model = TorchNetwork(network)
    # test default parameters
    assert type(model.criterion) is MSELoss, "Default loss wasn't of type MSELoss."
    assert type(model.optimizer) is Adam, "Default optimizer wasn't of type Adam."
    assert len(model.get_weights()) == 6, (
        "The number of weights doesn't match the expected value of 6."
    )
    assert type(model.activations) is dict, "Default activations wasn't of type dict."
    assert len(model.activations) == 0, "Default activations wasn't empty."

    # init model with parameters
    model = TorchNetwork(
        network, optimizer, loss, optimizer_params, loss_params, activations
    )
    # test model parameters
    assert type(model.criterion) is HuberLoss, "Model loss wasn't of type HuberLoss."
    assert model.criterion.delta == 1.1, (
        "Huber loss' delta parameter doesn't match expected value of 1.1."
    )
    assert type(model.optimizer) is Adam, "Model optimizer wasn't of type Adam."
    state_dict = model.optimizer.state_dict()
    assert state_dict['param_groups'][0]['lr'] == 0.005 and state_dict['param_groups'][
        0
    ]['betas'] == (0.91, 0.998), (
        "Model optimizer's parameters don't match the expected values"
        " (lr==0.005, betas==(0.91, 0.998))."
    )
    assert len(model.activations) == 3, (
        "Length of activations doesn't match the expected value of 3."
    )

    # test layer activity retrieval
    # with layer indices
    for i, size in enumerate([64, 64, 1]):
        layer_shape = model.get_layer_activity(data[:10], i).shape
        assert layer_shape == (size, 10), (
            "Shape of activity at layer %d didn't match the expected shape of (10, %d)."
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
            "Shape of activity at layer %s didn't match the expected shape of (10, %d)."
            % (layer, size)
        )

    # test training
    weights_before = model.get_weights()
    preds_before = model.predict_on_batch(data).flatten()
    error_before = np.mean((preds_before - labels) ** 2)
    for _ in range(100):
        for batch in range(20):
            start, end = batch * 50, (batch + 1) * 50
            model.train_on_batch(data[start:end], labels[start:end])
    preds_after = model.predict_on_batch(data).flatten()
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
    assert type(model.criterion) == type(model_cloned.criterion), (
        "Cloned model's loss function doesn't match."
    )
    assert model.criterion.delta == model_cloned.criterion.delta, (
        "Cloned model's loss function's delta parameter doesn't match."
    )
    assert type(model.optimizer) == type(model_cloned.optimizer), (
        "Cloned model's optimizer doesn't match."
    )
    param_group, param_group_cloned = (
        model.optimizer.state_dict()['param_groups'][0],
        model_cloned.optimizer.state_dict()['param_groups'][0],
    )
    assert (
        param_group['lr'] == param_group_cloned['lr']
        and param_group['betas'] == param_group_cloned['betas']
    ), "Cloned model's optimizer's parameter don't match."
    assert type(model.activations) == type(model_cloned.activations), (
        "Cloned model's activations doesn't match."
    )
    for i, d in enumerate(model_cloned.activations):
        idx = d if type(d) is str else i
        assert model.activations[idx] is model_cloned.activations[idx], ( # type: ignore
            "Cloned model's activation at layer %s doesn't match." % idx
        )
    assert model.model.parameters() != model_cloned.model.parameters(), (
        'Cloned model shares underlying parameters.'
    )

    # test setters
    # loss function
    model.set_loss('mse')
    assert type(model.criterion) is MSELoss, (
        "Loss function wasn't set when using string identifier."
    )
    model.set_loss('huber', loss_params)
    assert model.criterion.delta == 1.1, (
        "Loss function's delta parameter wasn't set when using string identifier."
    )
    model.set_loss('mse')
    model.set_loss(loss, loss_params)
    assert type(model.criterion) is HuberLoss, (
        "Loss function wasn't set when using loss class."
    )
    assert model.criterion.delta == 1.1, (
        "Loss function's delta parameter wasn't set when using loss class."
    )
    # optimizer
    model.set_optimizer('adam', {'lr': 0.1})
    assert type(model.optimizer) is Adam, (
        "Optimizer wasn't set when using string identifier"
    )
    assert model.optimizer.state_dict()['param_groups'][0]['lr'] == 0.1, (
        "Optimizer's lr parameter wasn't set when using string identifier."
    )
    model.set_optimizer(Adam(model.model.parameters()), {'lr': 0.01})
    assert type(model.optimizer) is Adam, (
        "Optimizer wasn't set when using string identifier"
    )
    assert model.optimizer.state_dict()['param_groups'][0]['lr'] == 0.01, (
        "Optimizer's lr parameter wasn't set when using string identifier."
    )

    # layer freezing
    # freeze and unfreeze all layers via name
    model.set_trainable(['layer_dense_1', 'layer_dense_2'], False)
    for l, w in zip(
            ['layer_dense_1', 'layer_dense_2'], ['weight', 'bias'], strict=False
            ):
        assert not model.model.get_parameter('%s.%s' % (l, w)).requires_grad
    model.set_trainable(['layer_dense_1', 'layer_dense_2'], True)
    for l, w in zip(
            ['layer_dense_1', 'layer_dense_2'], ['weight', 'bias'], strict=False
            ):
        assert model.model.get_parameter('%s.%s' % (l, w)).requires_grad
    # freeze and unfreeze all layers via index
    model.set_trainable([0, 1], False)
    for l, w in zip(
            ['layer_dense_1', 'layer_dense_2'], ['weight', 'bias'], strict=False
            ):
        assert not model.model.get_parameter('%s.%s' % (l, w)).requires_grad
    model.set_trainable([0, 1], True)
    for l, w in zip(
            ['layer_dense_1', 'layer_dense_2'], ['weight', 'bias'], strict=False
            ):
        assert model.model.get_parameter('%s.%s' % (l, w)).requires_grad
    # freeze and unfreeze specific layers via name
    model.set_trainable(
        ['layer_dense_1', 'layer_dense_2'],
        {'layer_dense_2': True, 'layer_dense_1': False},
    )
    assert not model.model.get_parameter('layer_dense_1.weight').requires_grad
    assert not model.model.get_parameter('layer_dense_1.bias').requires_grad
    assert model.model.get_parameter('layer_dense_2.weight').requires_grad
    assert model.model.get_parameter('layer_dense_2.bias').requires_grad
    model.set_trainable(['layer_dense_1', 'layer_dense_2'], True)
    for l, w in zip(
            ['layer_dense_1', 'layer_dense_2'], ['weight', 'bias'], strict=False
            ):
        assert model.model.get_parameter('%s.%s' % (l, w)).requires_grad
    # freeze and unfreeze specific layers via index
    model.set_trainable([0, 1], [False, True])
    assert not model.model.get_parameter('layer_dense_1.weight').requires_grad
    assert not model.model.get_parameter('layer_dense_1.bias').requires_grad
    assert model.model.get_parameter('layer_dense_2.weight').requires_grad
    assert model.model.get_parameter('layer_dense_2.bias').requires_grad
    model.set_trainable([0, 1], True)
    for l, w in zip(
            ['layer_dense_1', 'layer_dense_2'], ['weight', 'bias'], strict=False
            ):
        assert model.model.get_parameter('%s.%s' % (l, w)).requires_grad
