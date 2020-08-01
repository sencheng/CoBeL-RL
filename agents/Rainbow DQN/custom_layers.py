from tensorflow.keras.layers import Dense, InputSpec
from tensorflow.keras import initializers
from tensorflow.keras import backend as K

class NoisyDense(Dense):
    def __init__(self, units, **kwargs):
        self.output_dim = units
        super(NoisyDense, self).__init__(units, **kwargs)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        self.input_dim = input_shape[-1]

        self.kernel = self.add_weight(shape=(self.input_dim, self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=None,
                                      constraint=None)

        # Zweiter Kernel (trainable weights) für Steuerung des Zufalls.
        self.kernel_sigma = self.add_weight(shape=(self.input_dim, self.units),
                                      initializer=initializers.Constant(0.017),
                                      name='sigma_kernel',
                                      regularizer=None,
                                      constraint=None)

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=None,
                                        constraint=None)

            # trainable, Steuerung des Zufalls des Bias.
            self.bias_sigma = self.add_weight(shape=(self.units,),
                                        initializer=initializers.Constant(0.02),
                                        name='bias_sigma',
                                        regularizer=None,
                                        constraint=None)
        else:
            self.bias = None
            self.epsilon_bias = None
        
        self.kernel_epsilon = K.zeros(shape=(self.input_dim, self.units))
        self.epsilon_bias = K.zeros(shape=(self.units,))

        self.input_spec = InputSpec(min_ndim=2, axes={-1: self.input_dim})
        self.built = True

    def call(self, inputs):
        perturbation = self.kernel_sigma * self.kernel_epsilon
        perturbed_kernel = self.kernel + perturbation
        output = K.dot(inputs, perturbed_kernel)
        if self.use_bias:
            bias_perturbation = self.bias_sigma * self.epsilon_bias
            perturbed_bias = self.bias + bias_perturbation
            output = K.bias_add(output, perturbed_bias)
        if self.activation is not None:
            output = self.activation(output)
        return output