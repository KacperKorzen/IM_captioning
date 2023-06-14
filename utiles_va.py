import string
import sys
import os
from os import listdir
import numpy as np
import random
import pickle
import json

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.applications import VGG16, xception

from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img 

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

def print_progress(count, max_count):
    """function to print progress of current operation

    Args:
        count (int): current state
        max_count (int): last state
    """
    # Percentage completion.
    pct_complete = count / max_count

    # Status-message. Note the \r which means the line should
    # overwrite itself.
    msg = "\r- Progress: {0:.1%}".format(pct_complete)

    # Print it.
    sys.stdout.write(msg)
    sys.stdout.flush()

def extract_features_VGG16(directory: str):
    """extract_features_VGG16 is used to load end extract features of photos from dataset

    Args:
        directory (str): directory to path of images

    Returns:
        dictionary: output of image features
    """
    # load model VGG16

    model = VGG16(include_top = True)
    
    # pop last fully connect layer
    model = Model(inputs = model.input,
                outputs = model.layers[-5].output)

    print(model.summary())
    print('\n')
    # helper variable
    features = {}
    # extract features from each photo
    counter = 0
    file_list = listdir(directory)
    file_list.sort()
    if file_list[0] == '.DS_Store':
        file_list.pop(0)

    for name in file_list:
        #loading image
        file_path = directory + '/' + name
        image = load_img(file_path, target_size = (model.input.shape[1],model.input.shape[2]))
        # convert the image pixels to a numpy array
        image = img_to_array(image, dtype = 'float16')
        # reshape data for the model
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        # prepare the image for the VGG model
        image = preprocess_input(image)
        # get features

        feature = model(image)
        feature = tf.reshape(feature,
                             (feature.shape[0], -1, feature.shape[3]))

        # get image id
        image_id = name.split('.')[0]

        feature_path = directory[:-6]+ "features_vgg_va/" + str(image_id)+ '.npy'
        # store feature
        features[image_id] = feature_path#feature
        np.save(feature_path, feature.numpy())
        counter += 1
        print_progress(counter, len(listdir(directory)))
    print('\nTransforming images succeed')
    features_shape = feature.shape[2]
    attention_features_shape = feature.shape[1]
    return features, features_shape, attention_features_shape

def extract_features_xcepction(directory: str):
    """extract_features_xception is used to load end extract features of photos from dataset

    Args:
        directory (str): directory to path of images

    Returns:
        dictionary: output of image features
    """
    # load model Xception

    model = xception.Xception(include_top = True)
    
    # pop last fully connect layer
    model = Model(inputs = model.input,
                outputs = model.layers[-4].output)

    print(model.summary())
    print('\n')
    # helper variable
    features = {}
    # extract features from each photo
    counter = 0
    file_list = listdir(directory)
    file_list.sort()
    if file_list[0] == '.DS_Store':
        file_list.pop(0)

    for name in file_list:
        #loading image
        file_path = directory + '/' + name
        image = load_img(file_path, target_size = (model.input.shape[1],model.input.shape[2]))
        # convert the image pixels to a numpy array
        image = img_to_array(image, dtype = 'float16')
        # reshape data for the model
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        # prepare the image for the VGG model
        image = xception.preprocess_input(image)
        # get features

        feature = model(image)
        feature = tf.reshape(feature,
                             (feature.shape[0], -1, feature.shape[3]))

        # get image id
        image_id = name.split('.')[0]
        feature_path = directory[:-6]+ "features_xception_va/" + str(image_id)+ '.npy'
        # store feature
        features[image_id] = feature_path#feature
        np.save(feature_path, feature.numpy())
        counter += 1
        print_progress(counter, len(listdir(directory)))
    print('\nTransforming images succeed')
    features_shape = feature.shape[2]
    attention_features_shape = feature.shape[1]
    return features, features_shape, attention_features_shape


class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, features, hidden):
        # features(CNN_encoder output) shape == (batch_size, 64, embedding_dim)

        # hidden shape == (batch_size, hidden_size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden_size)
        hidden_with_time_axis = tf.expand_dims(hidden, 1)

        # attention_hidden_layer shape == (batch_size, 64, units)
        attention_hidden_layer = (tf.nn.tanh(self.W1(features) +
                                         self.W2(hidden_with_time_axis)))

        # score shape == (batch_size, 64, 1)
        # This gives you an unnormalized score for each image feature.
        score = self.V(attention_hidden_layer)

        # attention_weights shape == (batch_size, 64, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights
    
class CNN_Encoder(tf.keras.Model):
    # Since you have already extracted the features and dumped it
    # This encoder passes those features through a Fully connected layer
    def __init__(self, embedding_dim):
        super(CNN_Encoder, self).__init__()
        # shape after fc == (batch_size, 64, embedding_dim)
        self.fc = tf.keras.layers.Dense(embedding_dim)

    def call(self, x):
        x = self.fc(x)
        x = tf.nn.relu(x)
        return x
    
class RNN_Decoder(tf.keras.Model):
    def __init__(self, embedding_dim, units, vocab_size):
        super(RNN_Decoder, self).__init__()
        self.units = units

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        # GRU lub LSTM TODO dodać wybor
        self.gru = tf.keras.layers.GRU(self.units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc1 = tf.keras.layers.Dense(self.units)
        self.fc2 = tf.keras.layers.Dense(vocab_size)

        self.attention = BahdanauAttention(self.units)

    def call(self, x, features, hidden):
        # defining attention as a separate model
        context_vector, attention_weights = self.attention(features, hidden)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # passing the concatenated vector to the GRU
        output, state = self.gru(x)

        # shape == (batch_size, max_length, hidden_size)
        x = self.fc1(output)

        # x shape == (batch_size * max_length, hidden_size)
        x = tf.reshape(x, (-1, x.shape[2]))

        # output shape == (batch_size * max_length, vocab)
        x = self.fc2(x)

        return x, state, attention_weights

    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.units))
    
def batch_generator(batch_size, train_keys, train_images, train_captions, max_tokens):
    """
    Generator function for creating random batches of training-data.
    
    Note that it selects the data completely randomly for each
    batch, corresponding to sampling of the training-set with
    replacement. This means it is possible to sample the same
    data multiple times within a single epoch - and it is also
    possible that some data is not sampled at all within an epoch.
    However, all the data should be unique within a single batch.
    """

    # Infinite loop.
    while True:
        # Get a list of random indices for images in the training-set.
        idx = np.random.randint(len(train_images),
                                size=batch_size)
        
        key = np.array(train_keys)[idx]

        # Get the pre-computed transfer-values for those images.
        # These are the outputs of the pre-trained image-model.
        
        transfer_values = []
        for el in key:
            transfer_values.append(np.squeeze(train_images[el]))

        # For each of the randomly chosen images there are
        # at least 5 captions describing the contents of the image.
        # Select one of those captions at random and get the
        # associated sequence of integer-tokens.
        tokens = []
        for el in key:
            all_tokens_photo = train_captions[el]

            tokens.append(all_tokens_photo[random.randint(0, len(all_tokens_photo)-1)])

        
        # Pad all the other token-sequences with zeros
        # so they all have the same length and can be
        # input to the neural network as a numpy array.
        tokens_padded = pad_sequences(tokens,
                                      maxlen=max_tokens,
                                      padding='post',
                                      truncating='post')
        
        # Further prepare the token-sequences.
        # The decoder-part of the neural network
        # will try to map the token-sequences to
        # themselves shifted one time-step.
        decoder_input_data = tokens_padded[:, 0:-1]
        decoder_output_data = tokens_padded[:, 1:]

        # Dict for the input-data. Because we have
        # several inputs, we use a named dict to
        # ensure that the data is assigned correctly.
        x_data = \
        {
            'decoder_input': decoder_input_data,
            'transfer_values_input': np.array(transfer_values)
        }

        # Dict for the output-data.
        y_data = \
        {
            'decoder_output': decoder_output_data
        }
        
        yield (x_data, y_data)
