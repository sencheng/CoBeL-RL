import numpy as np
import tensorflow_probability as tfp
import tensorflow as tf
import tensorflow.keras.layers as tfkl
import tensorflow.keras as tfk
import matplotlib.pyplot as plt
tfd = tfp.distributions
tfb = tfp.bijectors

n_arr = np.random.rand(1000,16,64,4)

# Density estimation with MADE.
made = tfb.AutoregressiveNetwork(params=2, hidden_units=[32, 32])

distribution = tfd.TransformedDistribution(
    distribution=tfd.Normal(loc=0., scale=1.),
    bijector=tfb.MaskedAutoregressiveFlow(made),
    event_shape=[2])

# # Construct and fit model.

# in_ = tfkl.Input(shape=(16,64,4), dtype=tf.float32)
# conv1 = tfkl.Conv2D(4,3,1)(in_)
# mp1 = tfkl.MaxPool2D(2)(conv1)
# fl = tfkl.Flatten()(mp1)
# d = tfkl.Dense(128,activation="relu")(fl)
# out = tfkl.Dense(2)(d)
# log_prob_layer = distribution.log_prob(out)
# model = tfk.Model(in_, log_prob_layer)

# model.compile(optimizer=tf.optimizers.Adam(),
#               loss=lambda _, log_prob: -log_prob)


# x_new2 = distribution.sample((2000))
# plt.scatter(x_new2[:,0],x_new2[:,1])
# plt.show()

# batch_size = 100
# model.fit(x=n_arr,
#           y=np.zeros((1000, 0), dtype=np.float32),
#           batch_size=batch_size,
#           epochs=3,
#           steps_per_epoch=1000/batch_size,  # Usually `n // batch_size`.
#           shuffle=True,
#           verbose=False)

# # output = model(n_arr)
# # print(output)
# # print(output.shape)

# sample = distribution.sample((3, 1))
# distribution.log_prob(np.ones((3, 2), dtype=np.float32))


# # plt.scatter(x_new2[:,0],x_new2[:,1])
# # plt.show()




