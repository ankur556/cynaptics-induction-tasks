import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from math import sqrt
from numpy import load, asarray, zeros, ones, savez_compressed
from numpy.random import randn, randint
import tensorflow
print(tensorflow.config.list_physical_devices('GPU'))
tensorflow.config.run_functions_eagerly(True)
# tensorflow.config.run_functions_eagerly(True)
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    # Allow GPU memory growth (allocates memory as needed)
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
# from tensorflow import keras
from skimage.transform import resize
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape, Conv2D
from tensorflow.keras.layers import UpSampling2D, AveragePooling2D, LeakyReLU, Layer, Add
from keras.constraints import max_norm
from keras.initializers import RandomNormal
import mtcnn
from mtcnn.mtcnn import MTCNN
from keras import backend
from matplotlib import pyplot
import cv2
import os
from os import listdir
from PIL import Image
import cv2
# load the prepared dataset
from numpy import load
# data = load('img_align_celeba_128.npz')
# faces = data['arr_0']
# print('Loaded: ', faces.shape)
# Loading the image file
def load_image(filename):
    image = Image.open(filename)
    image = image.convert('RGB')
    pixels = asarray(image)
    return pixels

# extract the face from a loaded image and resize
def extract_face(model, pixels, required_size=(256,256)):
    # detect face in the image
    faces = model.detect_faces(pixels)
    if len(faces) == 0:
        return None

    # extract details of the face
    x1, y1, width, height = faces[0]['box']
    x1, y1 = abs(x1), abs(y1)

    x2, y2 = x1 + width, y1 + height
    face_pixels = pixels[y1:y2, x1:x2]
    image = Image.fromarray(face_pixels)
    image = image.resize(required_size)
    face_array = asarray(image)

    return face_array

# load images and extract faces for all images in a directory
def load_faces(directory, n_faces):
    # prepare model
    model = MTCNN()
    faces = list()

    for filename in os.listdir(directory):
        # Computing the retrieval and extraction of faces
        pixels = load_image(directory + filename)
        face = extract_face(model, pixels)
        if face is None:
            continue
        faces.append(face)
        # print(len(faces), face.shape)
        if len(faces) >= n_faces:
            break

    return asarray(faces)
# pixel-wise feature vector normalization layer
class PixelNormalization(Layer):
    # initialize the layer
    def __init__(self, **kwargs):
        super(PixelNormalization, self).__init__(**kwargs)

    # perform the operation
    def call(self, inputs):
        # computing pixel values
        values = inputs**2.0
        mean_values = backend.mean(values, axis=-1, keepdims=True)
        mean_values += 1.0e-8
        l2 = backend.sqrt(mean_values)
        normalized = inputs / l2
        return normalized

    # define the output shape of the layer
    def compute_output_shape(self, input_shape):
        return input_shape
# weighted sum output
# mini-batch standard deviation layer
class MinibatchStdev(Layer):
    # initialize the layer
    def __init__(self, **kwargs):
        super(MinibatchStdev, self).__init__(**kwargs)

    # perform the operation
    def call(self, inputs):
        mean = backend.mean(inputs, axis=0, keepdims=True)
        squ_diffs = backend.square(inputs - mean)
        mean_sq_diff = backend.mean(squ_diffs, axis=0, keepdims=True)
        mean_sq_diff += 1e-8
        stdev = backend.sqrt(mean_sq_diff)

        mean_pix = backend.mean(stdev, keepdims=True)
        shape = backend.shape(inputs)
        output = backend.tile(mean_pix, (shape[0], shape[1], shape[2], 1))

        combined = backend.concatenate([inputs, output], axis=-1)
        return combined

    # define the output shape of the layer
    def compute_output_shape(self, input_shape):
        input_shape = list(input_shape)
        input_shape[-1] += 1
        return tuple(input_shape)
#class WeightedSum(Add):
    # init with default value
    #def _init_(self, alpha=0.0, **kwargs):
        #super(WeightedSum, self)._init_(**kwargs)
        #self.alpha = backend.variable(alpha, name='ws_alpha')

    # output a weighted sum of inputs
    #def _merge_function(self, inputs):
        # only supports a weighted sum of two inputs
        #assert (len(inputs) == 2)
        # ((1-a) * input1) + (a * input2)
        #output = ((1.0 - self.alpha) * inputs[0]) + (self.alpha * inputs[1])
        #return output

class WeightedSum(Add):
    # init with default value
    def __init__(self, alpha=0.0, **kwargs):
        super(WeightedSum, self).__init__(**kwargs)
        self.alpha = backend.variable(alpha, name='ws_alpha')

    # output a weighted sum of inputs
    def _merge_function(self, inputs):
        # only supports a weighted sum of two inputs
        assert (len(inputs) == 2)
        # ((1-a) * input1) + (a * input2)
        output = ((1.0 - self.alpha) * inputs[0]) + (self.alpha * inputs[1])
        return output


# calculate wasserstein loss
def wasserstein_loss(y_true, y_pred):
    return backend.mean(y_true * y_pred)
# load dataset
def load_real_samples(filename):
    data = load(filename)
    X = data['arr_0']
    X = X.astype('float32')
    X = (X - 127.5) / 127.5
    return X

# select real samples
def generate_real_samples(dataset, n_samples):
    ix = randint(0, dataset.shape[0], n_samples)
    X = dataset[ix]
    y = ones((n_samples, 1))
    return X, y

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
    x_input = randn(latent_dim * n_samples)
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input

# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, latent_dim, n_samples):
    x_input = generate_latent_points(latent_dim, n_samples)
    X = generator.predict(x_input)
    y = -ones((n_samples, 1))
    return X, y

# update the alpha value on each instance of WeightedSum
def update_fadein(models, step, n_steps):
    alpha = step / float(n_steps - 1)
    for model in models:
        for layer in model.layers:
            if isinstance(layer, WeightedSum):
                backend.set_value(layer.alpha, alpha)
                # layer.alpha.assign(alpha)

# scale images to preferred size
def scale_dataset(images, new_shape):
    scaled_data = []
    for image in dataset:
        # Resize the image using TensorFlow
        new_image = tf.image.resize(image, new_shape)
        scaled_data.append(new_image)
    # return tf.stack(scaled_data)
    return tf.stack(scaled_data)
    # images_list = list()
    # for image in images:
    #     new_image = resize(image, new_shape, 0)
    #     images_list.append(new_image)
    # return asarray(images_list)
# adding a generator block
def add_generator_block(old_model):
    init = RandomNormal(stddev=0.02)
    const = max_norm(1.0)
    block_end = old_model.layers[-2].output

    # upsample, and define new block
    upsampling = UpSampling2D()(block_end)
    g = Conv2D(128, (3,3), padding='same', kernel_initializer=init, kernel_constraint=const)(upsampling)
    g = PixelNormalization()(g)
    g = LeakyReLU(alpha=0.2)(g)
    g = Conv2D(128, (3,3), padding='same', kernel_initializer=init, kernel_constraint=const)(g)
    g = PixelNormalization()(g)
    g = LeakyReLU(alpha=0.2)(g)

    out_image = Conv2D(3, (1,1), padding='same', kernel_initializer=init, kernel_constraint=const)(g)
    model1 = Model(old_model.input, out_image)
    out_old = old_model.layers[-1]
    out_image2 = out_old(upsampling)

    merged = WeightedSum()([out_image2, out_image])
    model2 = Model(old_model.input, merged)
    return [model1, model2]
# define generator models
def define_generator(latent_dim, n_blocks, in_dim=4):
    init = RandomNormal(stddev=0.02)
    const = max_norm(1.0)
    model_list = list()
    in_latent = Input(shape=(latent_dim,))
    g  = Dense(128 * in_dim * in_dim, kernel_initializer=init, kernel_constraint=const)(in_latent)
    g = Reshape((in_dim, in_dim, 128))(g)

    # conv 4x4, input block
    g = Conv2D(128, (3,3), padding='same', kernel_initializer=init, kernel_constraint=const)(g)
    g = PixelNormalization()(g)
    g = LeakyReLU(alpha=0.2)(g)

    # conv 3x3
    g = Conv2D(128, (3,3), padding='same', kernel_initializer=init, kernel_constraint=const)(g)
    g = PixelNormalization()(g)
    g = LeakyReLU(alpha=0.2)(g)

    # conv 1x1, output block
    out_image = Conv2D(3, (1,1), padding='same', kernel_initializer=init, kernel_constraint=const)(g)
    model = Model(in_latent, out_image)
    model_list.append([model, model])

    for i in range(1, n_blocks):
        old_model = model_list[i - 1][0]
        models = add_generator_block(old_model)
        model_list.append(models)

    return model_list
# adding a discriminator block
def add_discriminator_block(old_model, n_input_layers=3):
    init = RandomNormal(stddev=0.02)
    const = max_norm(1.0)
    in_shape = list(old_model.input.shape)

    # define new input shape as double the size
    input_shape = (in_shape[-2]*2, in_shape[-2]*2, in_shape[-1])
    in_image = Input(shape=input_shape)

    # define new input processing layer
    d = Conv2D(128, (1,1), padding='same', kernel_initializer=init, kernel_constraint=const)(in_image)
    d = LeakyReLU(alpha=0.2)(d)

    # define new block
    d = Conv2D(128, (3,3), padding='same', kernel_initializer=init, kernel_constraint=const)(d)
    d = LeakyReLU(alpha=0.2)(d)
    d = Conv2D(128, (3,3), padding='same', kernel_initializer=init, kernel_constraint=const)(d)
    d = LeakyReLU(alpha=0.2)(d)
    d = AveragePooling2D(pool_size=(2, 2))(d)
    block_new = d

    # skip the input, 1x1 and activation for the old model
    for i in range(n_input_layers, len(old_model.layers)):
        d = old_model.layers[i](d)
    model1 = Model(in_image, d)

    model1.compile(loss=wasserstein_loss, optimizer=Adam(learning_rate=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8))

    downsample = AveragePooling2D(pool_size=(2, 2))(in_image)

    block_old = old_model.layers[1](downsample)
    block_old = old_model.layers[2](block_old)
    d = WeightedSum()([block_old, block_new])

    for i in range(n_input_layers, len(old_model.layers)):
        d = old_model.layers[i](d)

    model2 = Model(in_image, d)

    model2.compile(loss=wasserstein_loss, optimizer=Adam(learning_rate=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8))
    return [model1, model2]
# define the discriminator models for each image resolution
def define_discriminator(n_blocks, input_shape=(4,4,3)):
    init = RandomNormal(stddev=0.02)
    const = max_norm(1.0)
    model_list = list()
    in_image = Input(shape=input_shape)

    d = Conv2D(128, (1,1), padding='same', kernel_initializer=init, kernel_constraint=const)(in_image)
    d = LeakyReLU(alpha=0.2)(d)
    d = MinibatchStdev()(d)

    d = Conv2D(128, (3,3), padding='same', kernel_initializer=init, kernel_constraint=const)(d)
    d = LeakyReLU(alpha=0.2)(d)
    d = Conv2D(128, (4,4), padding='same', kernel_initializer=init, kernel_constraint=const)(d)
    d = LeakyReLU(alpha=0.2)(d)

    d = Flatten()(d)
    out_class = Dense(1)(d)

    model = Model(in_image, out_class)
    model.compile(loss=wasserstein_loss, optimizer=Adam(learning_rate=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8))
    model_list.append([model, model])

    for i in range(1, n_blocks):
        old_model = model_list[i - 1][0]
        models = add_discriminator_block(old_model)
        model_list.append(models)

    return model_list

# define composite models for training generators via discriminators
def define_composite(discriminators, generators):
    model_list = list()
    # create composite models
    for i in range(len(discriminators)):
        g_models, d_models = generators[i], discriminators[i]
        # straight-through model
        d_models[0].trainable = False
        model1 = Sequential()
        model1.add(g_models[0])
        model1.add(d_models[0])
        model1.compile(loss=wasserstein_loss, optimizer=Adam(learning_rate=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8))
        # fade-in model
        d_models[1].trainable = False
        model2 = Sequential()
        model2.add(g_models[1])
        model2.add(d_models[1])
        model2.compile(loss=wasserstein_loss, optimizer=Adam(learning_rate=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8))
        # store
        model_list.append([model1, model2])
    return model_list
# train a generator and discriminator
def train_epochs(g_model, d_model, gan_model, dataset, n_epochs, n_batch, fadein=False):
    dataset = dataset.batch(n_batch).prefetch(tensorflow.data.AUTOTUNE).cache()
    bat_per_epo = len(dataset)  # Automatically determined from batching
    n_steps = bat_per_epo * n_epochs

    for epoch in range(n_epochs):
        # print(f"Epoch {epoch+1}/{n_epochs}")
        for step, real_images in enumerate(dataset):  # Loop through batches in the dataset
            if fadein:
                update_fadein([g_model, d_model, gan_model], epoch * bat_per_epo + step, n_steps)
            # Prepare real and fake samples
            half_batch = real_images.shape[0] // 2
            X_real, y_real = real_images[:half_batch], tensorflow.ones((half_batch, 1))
            X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)

            # Update the discriminator model
            d_loss1 = d_model.train_on_batch(X_real, y_real)
            d_loss2 = d_model.train_on_batch(X_fake, y_fake)

            # Update the generator via the discriminator's error
            z_input = generate_latent_points(latent_dim, n_batch)
            y_real2 = tensorflow.ones((n_batch, 1))
            g_loss = gan_model.train_on_batch(z_input, y_real2)

            # Summarize loss for this step
            print(f'>Step {step+1}/{bat_per_epo}, Epoch {epoch+1}/{n_epochs} d1={d_loss1:.3f}, d2={d_loss2:.3f}, g={g_loss:.3f}')

        # Optionally apply fade-in if specified


# train the generator and discriminator
def train(g_models, d_models, gan_models, dataset, latent_dim, e_norm, e_fadein, n_batch):
    print("Train begins...")
    g_normal, d_normal, gan_normal = g_models[0][0], d_models[0][0], gan_models[0][0]
    gen_shape = g_normal.output_shape
    scaled_data = scale_dataset(dataset, gen_shape[1:3])
    print('Scaled Data', scaled_data.shape)
    scaled_data = tensorflow.data.Dataset.from_tensor_slices(scaled_data)

    # train normal or straight-through models
    train_epochs(g_normal, d_normal, gan_normal, scaled_data, e_norm[0], n_batch[0])
    summarize_performance('tuned', g_normal,d_normal, latent_dim)

    # process each level of growth
    for i in range(1, len(g_models)):
        # retrieve models for this level of growth
        [g_normal, g_fadein] = g_models[i]
        [d_normal, d_fadein] = d_models[i]
        [gan_normal, gan_fadein] = gan_models[i]

        # scale dataset to appropriate size
        scaled_data = scale_dataset(dataset, gen_shape[1:3])
        print('Scaled Data', scaled_data.shape)
        scaled_data = tensorflow.data.Dataset.from_tensor_slices(scaled_data)

        # train fade-in models for next level of growth
        train_epochs(g_fadein, d_fadein, gan_fadein, scaled_data, e_fadein[i], n_batch[i], True)
        summarize_performance('faded', g_fadein,d_fadein, latent_dim)

        # train normal or straight-through models
        train_epochs(g_normal, d_normal, gan_normal, scaled_data, e_norm[i], n_batch[i])
        summarize_performance('tuned', g_normal,d_normal, latent_dim)
# generate samples and save as a plot and save the model
def summarize_performance(status, g_model,d_model, latent_dim, n_samples=25):
    gen_shape = g_model.output_shape
    name = '%03dx%03d-%s' % (gen_shape[1], gen_shape[2], status)

    X, _ = generate_fake_samples(g_model, latent_dim, n_samples)
    X = (X - X.min()) / (X.max() - X.min())

    square = int(sqrt(n_samples))
    for i in range(n_samples):
        pyplot.subplot(square, square, 1 + i)
        pyplot.axis('off')
        pyplot.imshow(X[i])

    # save plot to file
    filename1 = 'plot_%s.png' % (name)
    pyplot.savefig(filename1)
    pyplot.close()

    filename2 = 'gen_%s.h5' % (name)
    g_model.save(filename2)
    filename3 = 'discri_%s.h5' % (name)
    d_model.save(filename3)
    print('**Saved: %s , %s and %s ************' % (filename1, filename2,filename3))
# number of growth phases where 6 blocks == [4, 8, 16, 32, 64, 128]
# n_blocks = 6
dataset = load_real_samples('img_align_celeba_128.npz')
dataset = tensorflow.data.Dataset.from_tensor_slices(dataset)
# print('Loaded', dataset.shape)
print('Loaded', dataset.shape)

# mirrored_strategy = tensorflow.distribute.MirroredStrategy()

# with mirrored_strategy.scope():
n_blocks = 7
latent_dim = 100

d_models = define_discriminator(n_blocks)
g_models = define_generator(latent_dim, n_blocks)
gan_models = define_composite(d_models, g_models)

n_batch=[50,50,32,32,16,16,8]
n_epochs=[5,8,8,16,16,18,20]

# n_batch = [16, 16, 16, 8, 4, 4]
# n_epochs = [5, 8, 8, 10, 10, 10]
#n_batch = [50,50,40,40,24,24,8]
#n_epochs = [5, 8, 8, 10,10,15,20] cant use this due to system not being able to allocate enough memory
# n_batch = [1,1,1,1,1]
# n_epochs = [5, 8, 8, 10, 10]

train(g_models, d_models, gan_models, dataset, latent_dim, n_epochs, n_epochs, n_batch)