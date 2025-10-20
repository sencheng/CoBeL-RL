# basic imports
import numpy as np
# torch
from torch import Tensor, reshape, cat, set_num_threads
from torch.nn import Module, Linear, HuberLoss, MSELoss
from torch.optim import Adam
from torch.nn.functional import tanh, sigmoid
# CoBeL-RL
from cobel.network import FlexibleTorchNetwork
from cobel.network.network_torch import ParamDict


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


class ComplexModel(Module):
    def __init__(self, input_sizes: tuple, output_sizes: tuple):
        super().__init__()
        self.stream_in_1_1 = Linear(in_features=input_sizes[0], out_features=64)
        self.stream_in_1_2 = Linear(in_features=64, out_features=32)
        self.stream_in_2_1 = Linear(in_features=input_sizes[1], out_features=64)
        self.stream_in_2_2 = Linear(in_features=64, out_features=32)
        self.hidden = Linear(in_features=64, out_features=32)
        self.stream_out_1_1 = Linear(in_features=32, out_features=32)
        self.stream_out_1_2 = Linear(in_features=32, out_features=output_sizes[0])
        self.stream_out_2_1 = Linear(in_features=32, out_features=32)
        self.stream_out_2_2 = Linear(in_features=32, out_features=output_sizes[1])
        self.double()

    def forward(self, input_1: Tensor, input_2: Tensor) -> tuple[Tensor, Tensor]:
        # first stream
        stream_1 = self.stream_in_1_1(input_1)
        stream_1 = tanh(stream_1)
        stream_1 = self.stream_in_1_2(stream_1)
        stream_1 = tanh(stream_1)
        # second stream
        stream_2 = self.stream_in_2_1(input_2)
        stream_2 = tanh(stream_2)
        stream_2 = self.stream_in_2_2(stream_2)
        stream_2 = tanh(stream_2)
        # hidden representation
        hidden = cat((stream_1, stream_2), 1)
        hidden = self.hidden(hidden)
        hidden = tanh(hidden)
        # first stream
        stream_1 = self.stream_out_1_1(hidden)
        stream_1 = tanh(stream_1)
        stream_1 = self.stream_out_1_2(stream_1)
        stream_1 = sigmoid(stream_1)
        # second stream
        stream_2 = self.stream_out_2_1(hidden)
        stream_2 = tanh(stream_2)
        stream_2 = self.stream_out_2_2(stream_2)
        stream_2 = sigmoid(stream_2)

        return stream_1, stream_2


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
    loss_params: list[ParamDict] = [{'delta': 1.1}]
    optimizer = Adam(network.parameters())
    optimizer_params = {'lr': 0.005, 'betas': (0.91, 0.998)}
    activations = {
        'layer_dense_1': tanh,
        'layer_dense_2': tanh,
        'layer_output': sigmoid,
    }

    # init model with default parameters
    model = FlexibleTorchNetwork(network)
    # test default parameters
    assert type(model.criteria[0]) is MSELoss, "Default loss wasn't of type MSELoss."  # type: ignore
    assert type(model.optimizer) is Adam, "Default optimizer wasn't of type Adam."
    assert len(model.get_weights()) == 6, (
        "The number of weights doesn't match the expected value of 6."
    )
    assert type(model.activations) is dict, "Default activations wasn't of type dict."
    assert len(model.activations) == 0, "Default activations wasn't empty."

    # init model with parameters
    model = FlexibleTorchNetwork(
        network, optimizer, loss, optimizer_params, loss_params, [1], activations # type: ignore
    )
    # test model parameters
    assert type(model.criteria[0]) is HuberLoss, "Model loss wasn't of type HuberLoss."  # type: ignore
    assert model.criteria[0].delta == 1.1, ( # type: ignore
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
    assert type(model.criteria[0]) == type(model_cloned.criteria[0]), ( # type: ignore
        "Cloned model's loss function doesn't match."
    )
    assert model.criteria[0].delta == model_cloned.criteria[0].delta, ( # type: ignore
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
    assert type(model.criteria[0]) is MSELoss, ( # type: ignore
        "Loss function wasn't set when using string identifier."
    )
    model.set_loss('huber', loss_params)  # type: ignore
    assert model.criteria[0].delta == 1.1, ( # type: ignore
        "Loss function's delta parameter wasn't set when using string identifier."
    )
    model.set_loss('mse')
    model.set_loss(loss, loss_params)  # type: ignore
    assert type(model.criteria[0]) is HuberLoss, ( # type: ignore
        "Loss function wasn't set when using loss class."
    )
    assert model.criteria[0].delta == 1.1, ( # type: ignore
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

    # init network with multiple inputs and outputs
    network_complex = ComplexModel((2, 2), (1, 1))
    # prepare training hyper-parameters
    loss = [MSELoss(), HuberLoss()]  # type: ignore
    loss_params = [{}, {'delta': 1.1}]
    loss_weights = [0.5, 0.5]
    optimizer = Adam(network.parameters())
    optimizer_params = {'lr': 0.005, 'betas': (0.91, 0.998)}
    activations = {
        'stream_in_1_1': tanh,
        'stream_in_1_2': tanh,
        'stream_in_2_1': tanh,
        'stream_in_2_2': tanh,
        'hidden': tanh,
        'stream_out_1_1': tanh,
        'stream_out_1_2': sigmoid,
        'stream_out_2_1': tanh,
        'stream_out_2_2': sigmoid,
    }

    # init model with default parameters
    model = FlexibleTorchNetwork(network_complex)
    # test default parameters
    assert len(model.criteria) == 1, 'Incorrect number of default loss functions.'
    assert type(model.criteria[0]) is MSELoss, "Default loss wasn't of type MSELoss."  # type: ignore
    assert len(model.loss_weights) == 1, 'Wrong number of loss weights.'
    assert type(model.optimizer) is Adam, "Default optimizer wasn't of type Adam."
    assert len(model.get_weights()) == 18, (
        "The number of weights doesn't match the expected value of 6."
    )
    assert type(model.activations) is dict, "Default activations wasn't of type dict."
    assert len(model.activations) == 0, "Default activations wasn't empty."

    # init model with parameters
    model = FlexibleTorchNetwork(
        network_complex,
        optimizer,
        loss,
        optimizer_params,
        loss_params, # type: ignore
        loss_weights,
        activations,
    )
    # test parameters
    assert len(model.criteria) == 2, 'Wrong number of loss functions.'
    assert (
        type(model.criteria[0]) is MSELoss and type(model.criteria[1]) is HuberLoss  # type: ignore
    ), "Default loss wasn't of type MSELoss."
    assert model.criteria[1].delta == 1.1, (  # type: ignore
        "Huber loss' delta parameter doesn't match expected value of 1.1."
    )
    assert len(model.loss_weights) == 2, 'Wrong number of loss weights.'
    assert type(model.optimizer) is Adam, "Default optimizer wasn't of type Adam."
    state_dict = model.optimizer.state_dict()
    assert state_dict['param_groups'][0]['lr'] == 0.005 and state_dict['param_groups'][
        0
    ]['betas'] == (0.91, 0.998), (
        "Model optimizer's parameters don't match the expected values"
        " (lr==0.005, betas==(0.91, 0.998))."
    )
    assert len(model.get_weights()) == 18, (
        "The number of weights doesn't match the expected value of 18."
    )
    assert type(model.activations) is dict, "Default activations wasn't of type dict."
    assert len(model.activations) == 9, "Default activations wasn't empty."

    # test layer activity retrieval
    # with layer indices
    for i, size in enumerate([64, 32, 64, 32, 32, 32, 1, 32, 1]):
        layer_shape = model.get_layer_activity([data[:10], data[:10]], i).shape
        assert layer_shape == (size, 10), (
            "Shape of activity at layer %d didn't match the expected shape of (10, %d)."
            % (i, size)
        )
    # with layer names
    for layer, size in {
        'stream_in_1_1': 64,
        'stream_in_1_2': 32,
        'stream_in_2_1': 64,
        'stream_in_2_2': 32,
        'hidden': 32,
        'stream_out_1_1': 32,
        'stream_out_1_2': 1,
        'stream_out_2_1': 32,
        'stream_out_2_2': 1,
    }.items():
        layer_shape = model.get_layer_activity([data[:10], data[:10]], layer).shape
        assert layer_shape == (size, 10), (
            "Shape of activity at layer %s didn't match the expected shape of (10, %d)."
            % (layer, size)
        )

    # test prediction
    assert type(model.predict_on_batch([data[:5], data[:5]])) is list, (
        "Output type doesn't match the expected type (list)."
    )

    # test training
    weights_before = model.get_weights()
    preds_before = model.predict_on_batch([data, data])
    error_before = np.mean((preds_before[0].flatten() - labels) ** 2) + np.mean(  # type: ignore
        (preds_before[1].flatten() - labels) ** 2  # type: ignore
    )
    for _ in range(100):
        for batch in range(20):
            start, end = batch * 50, (batch + 1) * 50
            model.train_on_batch(
                [data[start:end], data[start:end]],
                [labels[start:end], labels[start:end]],
            )
    preds_after = model.predict_on_batch([data, data])
    error_after = np.mean((preds_after[0].flatten() - labels) ** 2) + np.mean(  # type: ignore
        (preds_after[1].flatten() - labels) ** 2  # type: ignore
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
    for i, criterion in enumerate(model.criteria):
        idx = criterion if type(criterion) is str else i
        assert type(model.criteria[idx]) is type(model_cloned.criteria[idx]), (  # type: ignore
            "Cloned model's loss function at index %s doesn't match." % idx
        )
    assert model.criteria[1].delta == model_cloned.criteria[1].delta, (  # type: ignore
        "Delta parameter of the cloned model's loss function at index 1 doesn't match."
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
        assert model.activations[idx] is model_cloned.activations[idx], (  # type: ignore
            "Cloned model's activation at layer %s doesn't match." % idx
        )
    assert model.model.parameters() != model_cloned.model.parameters(), (
        'Cloned model shares underlying parameters.'
    )

    # test setters
    # loss function
    model.set_loss(['huber', 'mse'])  # type: ignore
    assert (
        type(model.criteria[0]) is HuberLoss and type(model.criteria[1]) is MSELoss  # type: ignore
    ), "Loss functions weren't set when using string identifier."
    model.set_loss(['mse', 'huber'], loss_params)  # type: ignore
    assert model.criteria[1].delta == 1.1, (  # type: ignore
        "Loss function's delta parameter wasn't set when using string identifier."
    )
    model.set_loss(['mse', 'huber'], loss_params, [0.25, 0.75])  # type: ignore
    assert model.loss_weights == [0.25, 0.75], (
        "Loss weights weren't set when using string identifier."
    )
    model.set_loss(['huber', 'mse'])  # type: ignore
    model.set_loss(loss, loss_params)  # type: ignore
    assert type(model.criteria[1]) is HuberLoss, (  # type: ignore
        "Loss function weren't set when using loss class."
    )
    assert model.criteria[1].delta == 1.1, (  # type: ignore
        "Loss function's delta parameter wasn't set when using loss class."
    )
    model.set_loss(loss, loss_params, [0.15, 0.85])  # type: ignore
    assert model.loss_weights == [0.15, 0.85], (
        "Loss weights weren't set when using loss class."
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
    model.set_trainable(['stream_in_1_1', 'stream_in_1_2'], False)
    for l, w in zip(
            ['stream_in_1_1', 'stream_in_1_2'], ['weight', 'bias'], strict=False
            ):
        assert not model.model.get_parameter('%s.%s' % (l, w)).requires_grad
    model.set_trainable(['stream_in_1_1', 'stream_in_1_2'], True)
    for l, w in zip(
            ['stream_in_1_1', 'stream_in_1_2'], ['weight', 'bias'], strict=False
            ):
        assert model.model.get_parameter('%s.%s' % (l, w)).requires_grad
    # freeze and unfreeze all layers via index
    model.set_trainable([0, 1], False)
    for l, w in zip(
            ['stream_in_1_1', 'stream_in_1_2'], ['weight', 'bias'], strict=False
            ):
        assert not model.model.get_parameter('%s.%s' % (l, w)).requires_grad
    model.set_trainable([0, 1], True)
    for l, w in zip(
            ['stream_in_1_1', 'stream_in_1_2'], ['weight', 'bias'], strict=False
            ):
        assert model.model.get_parameter('%s.%s' % (l, w)).requires_grad
    # freeze and unfreeze specific layers via name
    model.set_trainable(
        ['stream_in_1_1', 'stream_in_1_2'],
        {'stream_in_1_2': True, 'stream_in_1_1': False},
    )
    assert not model.model.get_parameter('stream_in_1_1.weight').requires_grad
    assert not model.model.get_parameter('stream_in_1_1.bias').requires_grad
    assert model.model.get_parameter('stream_in_1_2.weight').requires_grad
    assert model.model.get_parameter('stream_in_1_2.bias').requires_grad
    model.set_trainable(['stream_in_1_1', 'stream_in_1_2'], True)
    for l, w in zip(
            ['stream_in_1_1', 'stream_in_1_2'], ['weight', 'bias'], strict=False
            ):
        assert model.model.get_parameter('%s.%s' % (l, w)).requires_grad
    # freeze and unfreeze specific layers via index
    model.set_trainable([0, 1], [False, True])
    assert not model.model.get_parameter('stream_in_1_1.weight').requires_grad
    assert not model.model.get_parameter('stream_in_1_1.bias').requires_grad
    assert model.model.get_parameter('stream_in_1_2.weight').requires_grad
    assert model.model.get_parameter('stream_in_1_2.bias').requires_grad
    model.set_trainable([0, 1], True)
    for l, w in zip(
            ['stream_in_1_1', 'stream_in_1_2'], ['weight', 'bias'], strict=False
            ):
        assert model.model.get_parameter('%s.%s' % (l, w)).requires_grad
