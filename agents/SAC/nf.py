import numpy as np
import tensorflow_probability as tfp
import tensorflow as tf
import tensorflow.keras.layers as tfkl
import tensorflow.keras as tfk
import matplotlib.pyplot as plt
tfd = tfp.distributions
tfb = tfp.bijectors 


n = 2000
x2 = np.random.randn(n).astype(dtype=np.float32) * 2.
x1 = np.random.randn(n).astype(dtype=np.float32) + (x2 * x2 / 4.)
data = np.stack([x1, x2], axis=-1)
print(data.shape)
plt.scatter(data[:,0],data[:,1])
plt.show()

# Density estimation with MADE.
made = tfb.AutoregressiveNetwork(params=2, hidden_units=[32, 32])

distribution = tfd.TransformedDistribution(
    distribution=tfd.Normal(loc=0., scale=1.),
    bijector=tfb.MaskedAutoregressiveFlow(made),
    event_shape=[2])

# Construct and fit model.
x_ = tfkl.Input(shape=(2,), dtype=tf.float32)
d1 = tfkl.Dense(64)(x_)
#...
log_prob_ = distribution.log_prob(x_)
model = tfk.Model(x_, log_prob_)

model.compile(optimizer=tf.optimizers.Adam(),
              loss=lambda _, log_prob: -log_prob)

batch_size = 100
model.fit(x=data,
          y=np.zeros((n, 0), dtype=np.float32),
          batch_size=batch_size,
          epochs=100,
          steps_per_epoch=n/batch_size,  # Usually `n // batch_size`.
          shuffle=True,
          verbose=False)

x_new2 = distribution.sample((2000))
plt.scatter(x_new2[:,0],x_new2[:,1])
plt.show()