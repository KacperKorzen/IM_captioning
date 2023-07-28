import string
import sys
import os
from os import listdir
import numpy as np
import random
import pickle
import json

from tensorflow.keras.models import Model
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.applications import VGG16, xception

from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img 

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

import nltk

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

def load_document(file_name):
    """load_document is a function used to read file contains captions

    Args:
        file_name (str): Absolute path to file contains captions of image

    Returns:
        str: Text of loaded file
    """
    # open the file as read only
    file = open(file_name, 'r')
    text = file.read()
    #close the file
    file.close()
    return text

def extract_description(document):
    """extract_description is a function used to extract description from loaded document

    Args:
        document (str): text from file contained description

    Returns:
        dictionary: dictionary of list contains file name and descriptions
    """
    mapping = dict()
    #process lines
    for line in document.split('\n'):
        #get tokens
        tokens = line.split()
        if len(line) < 2:
            continue
        image_id, image_caption = tokens[0], tokens[1:]
        # remove .jpg
        image_id = image_id.split('.')[0]
        # back tokens to string
        image_desc = " ".join(image_caption)
        # create list of captions
        if image_id not in mapping:
            mapping[image_id] = list()
        mapping[image_id].append(image_desc)

    if "image,caption" in mapping.keys():
        mapping.pop("image,caption")

    return mapping

def clean_captions(captions):
    """clean_caption is used to pre processing captions

    Args:
        captions (dict): dictionary contains captions

    Returns:
        dictionary: dictionary of list contains of file name and preprocessed descriptions
    """
    captions_out = {}
    table = str.maketrans('','',string.punctuation)
    for key, captions_list in captions.items():
        for i in range(len(captions_list)):
            caption = captions_list[i]
            # tokenize
            caption = caption.split()
            # convert to lower case
            caption = [word.lower() for word in caption]
			# remove punctuation from each token
            caption = [w.translate(table) for w in caption]
			# remove hanging 's' and 'a'
            caption = [word for word in caption if len(word)>1]
			# remove tokens with numbers in them
            caption = [word for word in caption if word.isalpha()]
            # remove tokens like an in the
            caption = [word for word in caption if (word != 'an' and word != 'the')]
            # add mark of begin and and
            caption = ['ssss'] + caption + ['eeee']
			# store as string
            captions_list[i] =  ' '.join(caption)
        captions_out[key] = captions_list 
    return captions_out

def flatten(captions):
    """flatten function is used to flatten dictionary of lists to tokenizer word

    Args:
        captions (dict): dictionary contains preprocessed captions

    Returns:
        list: flatten descriptions to tokenizer
    """
    captions_value = list(captions.values())

    flatten = []
    for element in captions_value:
        flatten += element
    
    return flatten


class TokenizerWrap(Tokenizer):
    """Wrap the Tokenizer-class from Keras with more functionality."""
    
    def __init__(self, texts, num_words=None):
        """
        :param texts: List of strings with the data-set.
        :param num_words: Max number of words to use.
        """

        Tokenizer.__init__(self, num_words=num_words, oov_token = "<uknw>")

        # Create the vocabulary from the texts.
        self.fit_on_texts(texts)

        # Create inverse lookup from integer-tokens to words.
        self.index_to_word = dict(zip(self.word_index.values(),
                                      self.word_index.keys()))

    def token_to_word(self, token):
        """Lookup a single word from an integer-token."""

        word = " " if token == 0 else self.index_to_word[token]
        return word 

    def tokens_to_string(self, tokens):
        """Convert a list of integer-tokens to a string."""

        # Create a list of the individual words.
        words = [self.index_to_word[token]
                 for token in tokens
                 if token != 0]
        
        # Concatenate the words to a single string
        # with space between all the words.
        text = " ".join(words)

        return text
    
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
                outputs = model.layers[-2].output)

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

        # get image id
        image_id = name.split('.')[0]

        feature_path = '/home2/Kacper_captioning/'+ "f8_features_vgg_classic/" + str(image_id)+ '.npy'
        # store feature
        features[image_id] = feature_path#feature
        np.save(feature_path, feature.numpy())
        counter += 1
        print_progress(counter, len(listdir(directory)))
    print('\nTransforming images succeed')
    return features

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

def load_records_coco(data_dir, train=True):
    """
    Load the image-filenames and captions
    for either the training-set or the validation-set.
    """

    if train:
        # Training-set.
        filename = "captions_train2017.json"
    else:
        # Validation-set.
        filename = "captions_val2017.json"

    # Full path for the data-file.
    path = os.path.join(data_dir, filename)

    # Load the file.
    with open(path, "r", encoding="utf-8") as file:
        data_raw = json.load(file)

    # Convenience variables.
    images = data_raw['images']
    annotations = data_raw['annotations']

    mapping = dict()
    #process lines
    max_count = len(images)
    count = 0
    for image in images:
        print_progress(count, max_count)
        image_name = image['file_name'][:-4]
        image_id = image['id']
        filtered_list = [ann for ann in annotations if ann['image_id'] == image_id]
        #filtered_list = filter(lambda x: x['image_id'] == image_id, annotations)
        # create list of captions
        if image_id not in mapping:
            mapping[image_name] = list()
        
        for el in filtered_list:
            mapping[image_name].append(el['caption'])
        count += 1

    return mapping

def extract_features_Xception(directory: str):
    """extract_features_VGG16 is used to load end extract features of photos from dataset

    Args:
        directory (str): directory to path of images

    Returns:
        dictionary: output of image features
    """
    # load model VGG16
    model = xception.Xception(include_top = True)
    
    # pop last fully connect layer
    model = Model(inputs = model.input,
                outputs = model.layers[-2].output)

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
        # prepare the image for the xception model
        image = xception.preprocess_input(image)
        # get features
        feature = model(image)

        # get image id
        image_id = name.split('.')[0]

        feature_path = '/home2/Kacper_captioning/'+ "f30_features_xception_classic/" + str(image_id)+ '.npy'
        # store feature
        features[image_id] = feature_path#feature
        np.save(feature_path, feature.numpy())
        counter += 1
        print_progress(counter, len(listdir(directory)))
    print('\nTransforming images succeed')
    return features
