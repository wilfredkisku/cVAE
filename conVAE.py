import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from IPython import display

import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import tensorflow_probability as tfp
import time

(train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()

#Get the train and test images 
print(train_images.shape)
print(test_images.shape)

def preprocessing_images(images):
    images = images.reshape((images.shape[0], 28, 28, 1))/255.
    return np.where(images > 0.5, 1.0, 0.0).astype('float32')

#reformat the images and also quantizing them into values 1.0 adn 0.0
train_images = preprocessing_images(train_images)
test_images = preprocessing_images(test_images)

train_size = 60000
batch_size = 32
test_size = 10000

#convert the dataset from tensor to slices, shuffle and convert to managable batch sizes 
#creating a input pipeline
train_dataset = (tf.data.Dataset.from_tensor_slices(train_images).shuffle(train_size).batch(batch_size))
test_dataset = (tf.data.Dataset.from_tensor_slices(test_images).shuffle(test_size).batch(batch_size))

class CVAE(tf.keras.Model):

    def __init__(self, latent_dim):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential(
                [
                    tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
                    tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2,2,), activation='relu'),
                    tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2,2,), activation='relu'),
                    tf.keras.layers.Flatten(),
                    tf.keras.layers.Dense(latent_dim + latent_dim),
                ]
                )
        self.decoder = tf.keras.Sequential(
                [
                    tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
                    tf.keras.layers.Dense(units=7*7*32, activation='relu'),
                    tf.keras.layers.Reshape(target_shape=(7,7,32)),
                    tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same', activation='relu'),
                    tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same', activation='relu'),
                    tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=3, strides=1, padding='same'),
                ]
                )
    
    #randomly sample epsilon for reparameteize 
    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    #encode pipeline
    def encode(self, x):
        mean, logvar = tf.split(self.encode(x), num_of_size_splits=2, axis=1)
        return mean, logvar
    
    #reparameterize 
    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.eps(logvar * 0.5) + mean
    
    #decode pipline
    def decode(self, z, apply_sigmoid=False):
        logits = self.decode(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits

optimizer = tf.keras.optimizers.Adam(1e-4)

def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2.*np.pi)
    return tf.reduce_sum(-.5 * ((sample-mean)**2 * tf.exp(-logvar) + logvar + log2pi), axis=raxis)

def compute_loss(model, x):
    #obtain the mean and logvar through the encoder
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)

    #obtain the logits in the higher dimension through decoder
    x_logit = model.decoder(z)
    
    #obtain the cross entropy loss
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)

    #obtain the logpx_z, logpz and logpz_x
    logpx_z = -tf.reduce_sum(cross_ent, axis=[1,2,3])
    logpz = log_normal_pdf(z, 0., 0.)
    logpz_x = log_normal_pdf(z, mean, logvar)
    return -tf.reduce_mean(logpx_z + logpz - logpz_x)

@tf.function
def train_step(model, x, optimizer):

    with tf.GradientTape() as tape:
        loss = compute_loss(model, x)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))



