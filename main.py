#%%
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import sys
import os
from PIL import Image
import pickle

import pandas as pd
import utiles 
from os import listdir
from sklearn.model_selection import train_test_split
import random

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Dense, GRU, Embedding, LSTM, add, Dropout
from tensorflow.keras.models import Model

from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

from tensorflow.keras.applications import VGG16

from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img 


#%%
PATH_FLIKER_8K_TOKENS = '/Users/korzeniewski/Desktop/IM_caption/dataset/FLICER8k/annotations/Flickr8k.token.txt'
PATH_FLIKER_30K_TOKENS = '/Users/korzeniewski/Desktop/IM_caption/dataset/FLICER30k/annotations/captions.txt'
PATH_COCO_TOKENS = '/Users/korzeniewski/Desktop/IM_caption/dataset/COCO/annotations'

PATH_FLIKER_8K_IMAGES = '/Users/korzeniewski/Desktop/IM_caption/dataset/FLICER8k/images'
PATH_FLIKER_30K_IMAGES = '/Users/korzeniewski/Desktop/IM_caption/dataset/FLICER30k/images'
PATH_COCO_IMAGES = '/Users/korzeniewski/Desktop/IM_caption/dataset/COCO/images'

COCO = True
NUM_WORDS = 10_000
MARK_START = 'ssss'
MARK_END = 'eeee'
SEED = 3
BATCH_SIZE = 256#128 #256
EMBEDDING_SIZE = 128
EPOCHS = 20

#%%
if COCO:
   captions = utiles.load_records_coco(PATH_COCO_TOKENS,train=1)
   with open('captions_coco_loaded_n.pickle', 'wb') as file:
        pickle.dump(captions, file)
        print('tu')
   captions = utiles.clean_captions(captions)
   with open('captions_coco_preprocesed_n.pickle', 'wb') as file2:
        pickle.dump(captions, file2)
else:
    captions = utiles.load_document(PATH_FLIKER_8K_TOKENS)
    captions = utiles.extract_description(captions)
    captions = utiles.clean_captions(captions)

#%%
flat = utiles.flatten(captions)
#flat = ' '.join(flat)

tokenizer = utiles.TokenizerWrap(texts=flat, num_words=NUM_WORDS)
token_start = tokenizer.word_index[MARK_START.strip()]
token_end = tokenizer.word_index[MARK_END.strip()]

tokens_captions = {}
for key in captions:
    token_captions = tokenizer.texts_to_sequences(captions[key])
    tokens_captions[key] = token_captions

# %%
if COCO:
    images_features = utiles.extract_features_VGG16(PATH_COCO_IMAGES+'/train2017')
else:
    images_features = utiles.extract_features_VGG16(PATH_FLIKER_8K_IMAGES)

#%%
images_names = list(images_features.keys())
train_keys, test_keys = train_test_split(images_names, random_state= SEED)

train_images = {key: images_features[key] for key in train_keys if key in images_features}
train_captions = {key: tokens_captions[key] for key in train_keys if key in tokens_captions}

test_images = {key: images_features[key] for key in test_keys if key in images_features}
test_captions = {key: tokens_captions[key] for key in test_keys if key in tokens_captions}

longest_list = []
CAPTION_LENGTH = 0

for key, value in train_captions.items():
    for inner_list in value:
        if len(inner_list) > CAPTION_LENGTH:
            CAPTION_LENGTH = len(inner_list)
            longest_list = inner_list

print("Longest list:", longest_list)
print("Longest list length:", CAPTION_LENGTH)

# %%
generator = utiles.batch_generator(BATCH_SIZE, train_keys, train_images, train_captions, CAPTION_LENGTH)
#%%
num_captions_train = 0

for el in train_captions:
        if isinstance(train_captions[el], list):
            num_captions_train += len(train_captions[el])

STEPS_PER_EPOCH = int(num_captions_train / BATCH_SIZE)

batch = next(generator)
batch_x = batch[0]
batch_y = batch[1]

# %%
TRANSFER_VALUE_SIZE = train_images[list(train_images.keys())[0]].shape[1]
STATE_SIZE = 512
VOCAB_SIZE = NUM_WORDS

'''
# Image
inputs1 = Input(shape=(TRANSFER_VALUE_SIZE,), name='transfer_values_input')
fe1 = Dropout(0.5)(inputs1)
fe2 = Dense(STATE_SIZE, activation='relu', name='decoder_transfer_map')(fe1)

# text
inputs2 = Input(shape=(CAPTION_LENGTH,), name = 'decoder_input')
se1 = Embedding(VOCAB_SIZE, EMBEDDING_SIZE,  name='decoder_embedding')(inputs2)
se2 = Dropout(0.5)(se1)
se3 = LSTM(STATE_SIZE)(se2)

# rest
decoder1 = add([fe2, se3])
decoder2 = Dense(STATE_SIZE, activation='relu')(decoder1)
outputs = Dense(VOCAB_SIZE, activation='softmax', name = 'decoder_output')(decoder2)

model = Model(inputs=[inputs1, inputs2], outputs=[outputs])
model.summary()
'''

transfer_values_input = Input(shape=(TRANSFER_VALUE_SIZE,),
                              name='transfer_values_input')
decoder_transfer_map = Dense(STATE_SIZE,
                             activation='tanh',
                             name='decoder_transfer_map')
decoder_input = Input(shape=(None, ), name='decoder_input') #None -> caption length
decoder_embedding = Embedding(input_dim=VOCAB_SIZE,
                              output_dim=EMBEDDING_SIZE,
                              name='decoder_embedding')
decoder_lstm = GRU(STATE_SIZE, name='decoder_lstm',
                   return_sequences=True)
decoder_dense = Dense(VOCAB_SIZE,
                      activation='softmax',
                      name='decoder_output')


initial_state = decoder_transfer_map(transfer_values_input)
# Start the decoder-network with its input-layer.
net = decoder_input
# Connect the embedding-layer.
net = decoder_embedding(net)
# Connect all the GRU layers.
net = decoder_lstm(net,initial_state=initial_state)
# Connect the final dense layer that converts to
# one-hot encoded arrays.
decoder_output = decoder_dense(net)

decoder_model = Model(inputs=[transfer_values_input, decoder_input],
                      outputs=[decoder_output])

decoder_model.compile(optimizer='adam',
                      loss='categorical_crossentropy')
decoder_model.compile(optimizer=RMSprop(lr=1e-3),
                      loss='sparse_categorical_crossentropy')
#%%
#save weights etc.
path_checkpoint = 'IM_caption_checkpoint_coco.keras'
callback_checkpoint = ModelCheckpoint(filepath=path_checkpoint,
                                      verbose=1,
                                      save_weights_only=True)
callback_tensorboard = TensorBoard(log_dir='./IM_caption_logs_coco/',
                                   histogram_freq=0,
                                   write_graph=False)
callbacks = [callback_checkpoint, callback_tensorboard]

#load weights
try:
    decoder_model.load_weights(path_checkpoint)
except Exception as error:
    print("Error trying to load checkpoint.")
    print(error)

decoder_model.fit(x=generator, steps_per_epoch=STEPS_PER_EPOCH, epochs=EPOCHS, callbacks=callbacks)

#%%
def generate_caption(image_path, file_name, max_tokens=CAPTION_LENGTH, model_cnn = 'VGG16', caption = 1, caption_path = PATH_FLIKER_8K_TOKENS):
    """
    Generate a caption for the image in the given path.
    The caption is limited to the given number of tokens (words).
    """
    if model_cnn == 'VGG16':
        # load model VGG16
        model = VGG16(include_top = True)
    
    # pop last fully connect layer
    model = Model(inputs = model.input,
                outputs = model.layers[-2].output)

    path = image_path + '/' + file_name + '.jpg'

    # Load and resize the image.
    image = load_img(path, target_size=(model.input.shape[1],model.input.shape[2]))

    # Expand the 3-dim numpy array to 4-dim
    # because the image-model expects a whole batch as input,
    # so we give it a batch with just one image.
    image_batch = np.expand_dims(image, axis=0)

    # Process the image with the pre-trained image-model
    # to get the transfer-values.
    transfer_values = model.predict(image_batch)

    # Pre-allocate the 2-dim array used as input to the decoder.
    # This holds just a single sequence of integer-tokens,
    # but the decoder-model expects a batch of sequences.
    shape = (1, max_tokens)
    decoder_input_data = np.zeros(shape=shape, dtype=np.int64)

    # The first input-token is the special start-token for 'ssss '.
    token_int = tokenizer.word_index[MARK_START.strip()]

    # Initialize an empty output-text.
    output_text = ''

    # Initialize the number of tokens we have processed.
    count_tokens = 0

    # While we haven't sampled the special end-token for ' eeee'
    # and we haven't processed the max number of tokens.
    while token_int != tokenizer.word_index[MARK_END.strip()] and count_tokens < max_tokens:
        # Update the input-sequence to the decoder
        # with the last token that was sampled.
        # In the first iteration this will set the
        # first element to the start-token.
        decoder_input_data[0, count_tokens] = token_int

        # Wrap the input-data in a dict for clarity and safety,
        # so we are sure we input the data in the right order.
        x_data = \
        {
            'transfer_values_input': transfer_values,
            'decoder_input': decoder_input_data
        }

        # Note that we input the entire sequence of tokens
        # to the decoder. This wastes a lot of computation
        # because we are only interested in the last input
        # and output. We could modify the code to return
        # the GRU-states when calling predict() and then
        # feeding these GRU-states as well the next time
        # we call predict(), but it would make the code
        # much more complicated.
        
        # Input this data to the decoder and get the predicted output.
        decoder_output = decoder_model.predict(x_data)

        # Get the last predicted token as a one-hot encoded array.
        # Note that this is not limited by softmax, but we just
        # need the index of the largest element so it doesn't matter.
        token_onehot = decoder_output[0, count_tokens, :]

        # Convert to an integer-token.
        token_int = np.argmax(token_onehot)

        # Lookup the word corresponding to this integer-token.
        sampled_word = tokenizer.token_to_word(token_int)

        # Append the word to the output-text.
        output_text += " " + sampled_word

        # Increment the token-counter.
        count_tokens += 1

    # This is the sequence of tokens output by the decoder.
    output_tokens = decoder_input_data[0]

    if caption:
        captions = utiles.load_document(caption_path)
        captions = utiles.extract_description(captions)

    # Plot the image.
    plt.imshow(image)
    plt.show()
    
    # Print the predicted caption.
    print("Predicted caption:")
    print(f"\t{output_text[:-5]}")
    print()
    if caption:
        print("Labels:")
        for el in captions[file_name]:
            print(f"\t{el}")
# %%
generate_caption(PATH_FLIKER_8K_IMAGES, random.choice(train_keys))
# %%

generate_caption('/Users/korzeniewski/Desktop/zdjecia/zdjecia_biznesowe',
                  'IMG_8062_tlo',
                  caption=0)
# %%
generate_caption(PATH_COCO_IMAGES+'/train2017', random.choice(train_keys), caption=0)

# %%

scorer_rouge = rouge_scorer.RougeScorer(['rouge1','rouge2', 'rougeL'], use_stemmer=True)

candidate_row = 'little girl is doing back bend in filed'
reference_row = 'little girl doing back bend'

candidate = candidate_row.split()
reference = reference_row.split()

# Calculate BLEU-1 score
bleu_1 = sentence_bleu([reference], candidate, weights=(1, 0, 0, 0))
# Calculate BLEU-2 score
bleu_2 = sentence_bleu([reference], candidate, weights=(0.5, 0.5, 0, 0))
# Calculate BLEU-3 score
bleu_3 = sentence_bleu([reference], candidate, weights=(1/3, 1/3, 1/3, 0))
# Calculate BLEU-4 score
bleu_4 = sentence_bleu([reference], candidate, weights=(0.25, 0.25, 0.25, 0.25))
# calculate Meteor
meteor = meteor_score([reference], candidate)
# rouge score
rouge_scores = scorer_rouge.score(reference_row, candidate_row)
rouge_1 = rouge_scores['rouge1'].fmeasure
rouge_2 = rouge_scores['rouge2'].fmeasure
rouge_L = rouge_scores['rougeL'].fmeasure
print("BLEU-1:", bleu_1)
print("BLEU-2:", bleu_2)
print("BLEU-3:", bleu_3)
print("BLEU-4:", bleu_4)
print("METEOR:", meteor)
print("ROUGE-1:", rouge_1)
print("ROUGE-2:", rouge_2)
print("ROUGE-L:", rouge_L)