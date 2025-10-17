import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import os

print("Libraries imported successfully!")
print(f"TensorFlow Version: {tf.__version__}")

# Suppress TensorFlow informational messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# --- 2. Load and Prepare the MNIST Dataset ---
print("\n--- Loading and Preparing MNIST Data ---")
# We only need the images (X) for this unsupervised task.
(x_train, _), (_, _) = keras.datasets.mnist.load_data()

# Preprocess the data for a GAN
# 1. Add a channel dimension (for Conv2D layers).
# 2. Normalize pixel values to the [-1, 1] range, which is standard for GANs with a tanh activation.
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype("float32")
x_train = (x_train - 127.5) / 127.5

# Create a TensorFlow Dataset for efficient batching
BUFFER_SIZE = 60000
BATCH_SIZE = 256
train_dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
print(f"Data loaded. Shape of training data: {x_train.shape}")


# --- 3. Define the GAN Components (Generator and Discriminator) ---

LATENT_DIM = 128

# The Generator model (The Art Forger)
def make_generator_model():
    model = keras.Sequential(name="generator")
    model.add(layers.Input(shape=(LATENT_DIM,)))
    model.add(layers.Dense(7 * 7 * 256, use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Reshape((7, 7, 256)))
    
    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(0.2))
    
    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(0.2))
    
    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    return model

# The Discriminator model (The Art Critic)
def make_discriminator_model():
    model = keras.Sequential(name="discriminator")
    model.add(layers.Input(shape=(28, 28, 1)))
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Dropout(0.3))
    
    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Dropout(0.3))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(1)) # No activation, outputs a raw score (logit)
    return model

generator = make_generator_model()
discriminator = make_discriminator_model()

print("\nGenerator and Discriminator models built.")


# --- 4. Define the WGAN-GP Model with Custom Training Logic ---

class WGAN(keras.Model):
    def __init__(self, discriminator, generator, latent_dim, discriminator_extra_steps=5, gp_weight=10.0):
        super().__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.d_steps = discriminator_extra_steps
        self.gp_weight = gp_weight

    def compile(self, d_optimizer, g_optimizer, d_loss_fn, g_loss_fn):
        super().compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn

    def gradient_penalty(self, batch_size, real_images, fake_images):
        alpha = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            pred = self.discriminator(interpolated, training=True)
        
        grads = gp_tape.gradient(pred, [interpolated])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def train_step(self, real_images):
        batch_size = tf.shape(real_images)[0]
        
        for i in range(self.d_steps):
            random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
            with tf.GradientTape() as tape:
                fake_images = self.generator(random_latent_vectors, training=True)
                fake_logits = self.discriminator(fake_images, training=True)
                real_logits = self.discriminator(real_images, training=True)
                
                d_cost = self.d_loss_fn(real_img=real_logits, fake_img=fake_logits)
                gp = self.gradient_penalty(batch_size, real_images, fake_images)
                d_loss = d_cost + gp * self.gp_weight
            
            d_gradient = tape.gradient(d_loss, self.discriminator.trainable_variables)
            self.d_optimizer.apply_gradients(zip(d_gradient, self.discriminator.trainable_variables))
            
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        with tf.GradientTape() as tape:
            generated_images = self.generator(random_latent_vectors, training=True)
            gen_img_logits = self.discriminator(generated_images, training=True)
            g_loss = self.g_loss_fn(gen_img_logits)
            
        gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(gen_gradient, self.generator.trainable_variables))
        
        return {"d_loss": d_loss, "g_loss": g_loss}

# Create a callback to save generated images during training
class GANMonitor(keras.callbacks.Callback):
    def __init__(self, num_img=16, latent_dim=128):
        self.num_img = num_img
        self.latent_dim = latent_dim
        self.seed = tf.random.normal([num_img, latent_dim])

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 5 == 0 or (epoch + 1) == EPOCHS:
            predictions = self.model.generator(self.seed, training=False)
            fig = plt.figure(figsize=(8, 8))
            
            for i in range(self.num_img):
                plt.subplot(4, 4, i + 1)
                plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
                plt.axis('off')
            
            plt.savefig(f'generated_image_epoch_{epoch+1:03d}.png')
            print(f'\nSaved sample images for epoch {epoch+1}.')
            plt.close(fig)

# --- 5. Compile and Train the WGAN-GP ---
print("\n--- Training the WGAN-GP (this will take a long time)... ---")
EPOCHS = 50

# Instantiate optimizers for the two models
generator_optimizer = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)
discriminator_optimizer = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)

# Define the loss functions
def discriminator_loss(real_img, fake_img):
    real_loss = tf.reduce_mean(real_img)
    fake_loss = tf.reduce_mean(fake_img)
    return fake_loss - real_loss

def generator_loss(fake_img):
    return -tf.reduce_mean(fake_img)

# Instantiate and compile the GAN model
wgan = WGAN(
    discriminator=discriminator,
    generator=generator,
    latent_dim=LATENT_DIM
)
wgan.compile(
    d_optimizer=discriminator_optimizer,
    g_optimizer=generator_optimizer,
    g_loss_fn=generator_loss,
    d_loss_fn=discriminator_loss,
)

# Train the model
wgan.fit(
    train_dataset, 
    epochs=EPOCHS, 
    callbacks=[GANMonitor(num_img=16, latent_dim=LATENT_DIM)]
)

print("\nTraining complete.")

# --- 6. Generate and Visualize Final Results ---
print("\n--- Generating Final Grid of Digits ---")
final_seed = tf.random.normal([100, LATENT_DIM])
final_predictions = wgan.generator.predict(final_seed)

fig = plt.figure(figsize=(10, 10))
for i in range(100):
    plt.subplot(10, 10, i + 1)
    plt.imshow(final_predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
    plt.axis('off')

plt.suptitle("Final AI-Generated Digits from WGAN-GP", fontsize=16)
plt.savefig('final_generated_digits_grid.png')
print("\nFinal grid of generated images saved as 'final_generated_digits_grid.png'")
plt.show()