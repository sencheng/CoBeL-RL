# basic imports
import numpy as np
# tensorflow
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()
from tensorflow import keras
from tensorflow.keras.layers import Dense, concatenate
# CoBel-RL framework
from cobel.networks.network_tensorflow import SequentialKerasNetwork, FunctionalKerasNetwork


def build_model(input_shape, number_of_actions):
    '''
    This function builds a simple network model. 
    
    Parameters
    ----------
    input_shape :                       The network model's input shape.
    number_of_actions :                 The network model's number of output units.
    
    Returns
    ----------
    model :                             The built network model.
    '''
    layer_input = keras.Input(shape=input_shape, name='layer_input')
    layer_dense_1 = Dense(64, activation='tanh', name='layer_dense_1')(layer_input)
    layer_dense_2 = Dense(64, activation='tanh', name='layer_dense_2')(layer_dense_1)
    layer_dense_3 = Dense(number_of_actions, activation='sigmoid', name='layer_output')(layer_dense_2)
    model = keras.Model(inputs=layer_input, outputs=layer_dense_3, name='simple_model')
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    
    return model

def build_model_multi(input_shape, number_of_actions):
    '''
    This function builds a simple network model. 
    
    Parameters
    ----------
    input_shape :                       The network model's input shape.
    number_of_actions :                 The network model's number of output units.
    
    Returns
    ----------
    model :                             The built network model.
    '''
    layer_input_1 = keras.Input(shape=input_shape, name='layer_input_1')
    layer_input_2 = keras.Input(shape=input_shape, name='layer_input_2')
    layer_concatenate = concatenate([layer_input_1, layer_input_2], name='layer_concatenate')
    layer_dense_1 = Dense(64, activation='tanh', name='layer_dense_1')(layer_concatenate)
    layer_dense_2 = Dense(64, activation='tanh', name='layer_dense_2')(layer_dense_1)
    layer_dense_3 = Dense(number_of_actions, activation='sigmoid', name='layer_output_1')(layer_dense_2)
    layer_dense_4 = Dense(number_of_actions, activation='sigmoid', name='layer_output_2')(layer_dense_2)
    model = keras.Model(inputs=[layer_input_1, layer_input_2], outputs=[layer_dense_3, layer_dense_4], name='simple_model')
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    
    return model

if __name__ == '__main__':
    # prepare observations
    observations = np.random.rand(5, 25)
    
    # use SequentialKerasNetwork class for simple (i.e., single input and single output) networks
    model = SequentialKerasNetwork(build_model((25,), 1))
    # use set_optimizer and and set_loss functions for setting the optimizer and loss function (accept optimizer/loss name or object as input)
    model.set_optimizer('adam')
    model.set_loss('mse')
    # predict_on_batch and train_on_batch functions for inference and training
    preds_1 = model.predict_on_batch(observations)
    for i in range(100):
        model.train_on_batch(observations, np.ones(5))
    # retrieve layer activity for a set of observations by specifing either layer name or layer index
    act_1 = model.get_layer_activity(observations, 'layer_output')
    act_2 = model.get_layer_activity(observations, -1)
    print(np.array_equal(act_1, act_2))
    
    # use FunctionalKerasNetwork class for complex networks
    model_2 = FunctionalKerasNetwork(build_model_multi((25,), 1))
    # in case of multiple outputs the loss function and its weighting can be set for each output 
    model_2.set_loss(['mse', 'mse'], loss_weights=[0.5, 0.5])
    # multiple inputs/outputs can be provided as list, dict or numpy.ndarray of object dtype
    preds_2 = model_2.predict_on_batch([observations, observations])
    for i in range(100):
        model_2.train_on_batch([observations, observations], [np.ones(5), np.zeros(5)])
    act_3 = model_2.get_layer_activity([observations, observations], 'layer_dense_2')
    # when using dicts the input keys must match the names of the input layers
    act_4 = model_2.get_layer_activity({'layer_input_1': observations, 'layer_input_2': observations}, 'layer_dense_2')
    # when using dicts the output keys must match the names of the output layers
    model_2.train_on_batch({'layer_input_1': observations, 'layer_input_2': observations}, {'layer_output_1': np.ones(5), 'layer_output_2': np.zeros(5)})
    model_2.set_loss({'layer_output_1': 'mse', 'layer_output_2': 'mse'}, loss_weights={'layer_output_1': 0.5, 'layer_output_2': 0.5})
    print(np.array_equal(act_3, act_4))
    
    # FunctionalKerasNetwork class also supports simple networks
    model_3 = FunctionalKerasNetwork(build_model((25,), 1))
    act_5 = model_3.get_layer_activity(observations, 'layer_output')
    act_6 = model_3.get_layer_activity([observations], 'layer_output')
    act_7 = model_3.get_layer_activity({'layer_input': observations}, 'layer_output')
    print(np.array_equal(act_5, act_6) and np.array_equal(act_6, act_7))
    
