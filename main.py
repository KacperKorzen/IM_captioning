#%%
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from nltk.translate.bleu_score import SmoothingFunction

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
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

from tensorflow.keras.preprocessing.image import load_img 
from tensorflow.keras.preprocessing.sequence import pad_sequences

#%%
PATH_FLIKER_8K_TOKENS = '/Users/korzeniewski/Desktop/IM_caption/dataset/FLICER8k/annotations/Flickr8k.token.txt'
PATH_FLIKER_30K_TOKENS = '/Users/korzeniewski/Desktop/IM_caption/dataset/FLICER30k/annotations/captions.txt'
PATH_COCO_TOKENS = '/Users/korzeniewski/Desktop/IM_caption/dataset/COCO/annotations'

PATH_FLIKER_8K_IMAGES = '/Users/korzeniewski/Desktop/IM_caption/dataset/FLICER8k/images'
PATH_FLIKER_30K_IMAGES = '/Users/korzeniewski/Desktop/IM_caption/dataset/FLICER30k/images'
PATH_COCO_IMAGES = '/Users/korzeniewski/Desktop/IM_caption/dataset/COCO/images'

COCO = False
NUM_WORDS = 10_000
MARK_START = 'ssss'
MARK_END = 'eeee'
SEED = 3
BATCH_SIZE = 32#128 #256
EMBEDDING_SIZE = 128
EPOCHS = 40

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

#########################################
#########################################
#########################################
#########################################
# %%
if COCO:
    images_features = utiles.extract_features_VGG16(PATH_COCO_IMAGES+'/train2017')
else:
    images_features = utiles.extract_features_VGG16(PATH_FLIKER_8K_IMAGES)
    #images_features= utiles.extract_features_xcepction(PATH_FLIKER_8K_IMAGES)
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

path_to_save = PATH_FLIKER_8K_IMAGES[:-6] + 'vgg_classic_'
with open(path_to_save + 'train_caption.pkl', 'wb') as fp:
    pickle.dump(train_captions, fp)
with open(path_to_save + 'train_images.pkl', 'wb') as fp:
    pickle.dump(train_images, fp)

with open(path_to_save + 'test_caption.pkl', 'wb') as fp:
    pickle.dump(test_captions, fp)
with open(path_to_save + 'test_images.pkl', 'wb') as fp:
    pickle.dump(test_images, fp)
# %%
#generator = utiles.batch_generator(BATCH_SIZE, train_keys, train_images, train_captions, CAPTION_LENGTH)
#%%
#num_captions_train = 0

#for el in train_captions:
#        if isinstance(train_captions[el], list):
#            num_captions_train += len(train_captions[el])

#STEPS_PER_EPOCH = int(num_captions_train / BATCH_SIZE)

#batch = next(generator)
#batch_x = batch[0]
#batch_y = batch[1]

# %%
#########################################
#########################################
#########################################
#########################################

path_to_save = PATH_FLIKER_8K_IMAGES[:-6] + 'vgg_classic_'#'xception_'
with open(path_to_save + 'train_caption.pkl', 'rb') as fp:
    train_captions = pickle.load(fp)
with open(path_to_save + 'train_images.pkl', 'rb') as fp:
    train_images = pickle.load(fp)
with open(path_to_save + 'test_caption.pkl', 'rb') as fp:
    test_captions = pickle.load(fp)
with open(path_to_save + 'test_images.pkl', 'rb') as fp:
    test_images = pickle.load(fp)

shapes = np.load(train_images[list(train_images.keys())[0]]).shape
TRANSFER_VALUE_SIZE = shapes[1]
STATE_SIZE = 512
BUFFER_SIZE = 1000
VOCAB_SIZE = NUM_WORDS

CAPTION_LENGTH = 0
for key, value in train_captions.items():
    for inner_list in value:
        if len(inner_list) > CAPTION_LENGTH:
            CAPTION_LENGTH = len(inner_list)


# training
captions_dataset_train = []
image_dataset_train = []

def map_func(img_name, cap):
    img_tensor = np.load(img_name)
    img_tensor = np.reshape(img_tensor, [img_tensor.shape[1]])
    return img_tensor, cap

train_keys = list(train_images.keys())
for key in train_keys:
    for i in range(len(train_captions[key])):
        image_dataset_train.append(train_images[key])
        captions_dataset_train.append(train_captions[key][i])
    #image_dataset_train.append(tf.reshape(train_images[key], [train_images[key].shape[1], train_images[key].shape[2]]))
    #captions_dataset_train.append(train_captions[key][np.random.randint(0,5)]) #moze nawet 6

tokens_padded = pad_sequences(captions_dataset_train, maxlen=CAPTION_LENGTH, padding='post',truncating='post')

dataset = tf.data.Dataset.from_tensor_slices((image_dataset_train, tokens_padded))

# Use map to load the numpy files in parallel
dataset = dataset.map(lambda item1, item2: tf.numpy_function(
          map_func, [item1, item2], [tf.float32, tf.int32]),
          num_parallel_calls=tf.data.AUTOTUNE)

# Shuffle and batch
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

# validation
captions_dataset_val = []
image_dataset_val = []

test_keys = list(test_images.keys())

for key in test_keys:
    for i in range(len(test_captions[key])):
        image_dataset_val.append(test_images[key])
        captions_dataset_val.append(test_captions[key][i])
    #image_dataset_val.append(tf.reshape(test_images[key], [test_images[key].shape[1], test_images[key].shape[2]]))
    #captions_dataset_val.append(test_captions[key][np.random.randint(0,5)])

tokens_padded_val = pad_sequences(captions_dataset_val, maxlen=CAPTION_LENGTH, padding='post',truncating='post')

dataset_val = tf.data.Dataset.from_tensor_slices((image_dataset_val, tokens_padded_val))

# Use map to load the numpy files in parallel
dataset_val = dataset_val.map(lambda item1, item2: tf.numpy_function(
          map_func, [item1, item2], [tf.float32, tf.int64]),
          num_parallel_calls=tf.data.AUTOTUNE)

# Shuffle and batch
dataset_val = dataset_val.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
dataset_val = dataset_val.prefetch(buffer_size=tf.data.AUTOTUNE)

num_steps_train = len(image_dataset_train) // BATCH_SIZE
num_steps_val = len(image_dataset_val) // BATCH_SIZE

STEPS_PER_EPOCH = len(image_dataset_train) // BATCH_SIZE
#%%

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
                      loss='sparse_categorical_crossentropy')
#decoder_model.compile(optimizer=RMSprop(lr=1e-3), loss='sparse_categorical_crossentropy')
#%%
#save weights etc.
'''
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
'''

def generator_fn(dataset):
    for item in dataset:
        img_tensor, target = item[0], item[1]

        decoder_input_data = target[:, 0:-1]
        decoder_output_data = target[:, 1:]

        # Dict for the input-data. Because we have several inputs, we use a named dict to
        # ensure that the data is assigned correctly.
        x_data = {
            'decoder_input': np.array(decoder_input_data),
            'transfer_values_input': np.array(img_tensor)}

        # Dict for the output-data.
        y_data = {'decoder_output': np.array(decoder_output_data)}
        yield (x_data, y_data)

start_epoch = 0
for epoch in range(start_epoch, EPOCHS):
    generator = generator_fn(dataset)

    decoder_model.fit(x=generator, epochs = 1, steps_per_epoch = STEPS_PER_EPOCH)

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

    while token_int != tokenizer.word_index[MARK_END.strip()] and count_tokens < max_tokens:
        # Update the input-sequence to the decoder
        # with the last token that was sampled.
        # In the first iteration this will set the
        # first element to the start-token.
        decoder_input_data[0, count_tokens] = token_int

        # Wrap the input-data in a dict for clarity and safety,
        # so we are sure we input the data in the right order.
        x_data = {
            'transfer_values_input': transfer_values,
            'decoder_input': decoder_input_data}
        
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

# %% metrics evaluattion
bleu_1_train = []; bleu_2_train = []; bleu_3_train = []; bleu_4_train = []
meteor_train = []
rouge_1_train = []; rouge_2_train = []; rouge_L_train = []
cider_train = []
scorer_rouge = rouge_scorer.RougeScorer(['rouge1','rouge2', 'rougeL'], use_stemmer=True)
chencherry = SmoothingFunction()

image_dataset_train_numpy = np.array(image_dataset_train)
captions_dataset_train_numpy = np.array(captions_dataset_train)
images = set(image_dataset_train)

counter = 0
for element in images:
    utiles.print_progress(counter, len(images))

    index = np.where(image_dataset_train_numpy == element)
    # data
    image_val = np.load(element)
    reference_captions = captions_dataset_train_numpy[index]
    caption_val_raw = []
    caption_val_split = []
    for el in reference_captions: 
        cap = tokenizer.tokens_to_string(el[1:-1])
        caption_val_raw.append(cap)
        caption_val_split.append(cap.split())
    
    # make prediction
    shape = (1, CAPTION_LENGTH)
    decoder_input_data = np.zeros(shape=shape, dtype=np.int64)

    # The first input-token is the special start-token for 'ssss '.
    token_int = tokenizer.word_index[MARK_START.strip()]

    # Initialize an empty output-text.
    output_text = ''

    # Initialize the number of tokens we have processed.
    count_tokens = 0

    while token_int != tokenizer.word_index[MARK_END.strip()] and count_tokens < CAPTION_LENGTH:
        # Update the input-sequence to the decoder
        # with the last token that was sampled.
        # In the first iteration this will set the
        # first element to the start-token.
        decoder_input_data[0, count_tokens] = token_int

        # Wrap the input-data in a dict for clarity and safety,
        # so we are sure we input the data in the right order.
        x_data = {
            'transfer_values_input': image_val,
            'decoder_input': decoder_input_data}
        
        # Input this data to the decoder and get the predicted output.
        decoder_output = decoder_model.predict(x_data, verbose=0)

        # Get the last predicted token as a one-hot encoded array.
        # Note that this is not limited by softmax, but we just
        # need the index of the largest element so it doesn't matter.
        token_onehot = decoder_output[0, count_tokens, :]

        # Convert to an integer-token.
        token_int = np.argmax(token_onehot)

        # Lookup the word corresponding to this integer-token.
        sampled_word = tokenizer.token_to_word(token_int)

        # Append the word to the output-text.
        if sampled_word != token_end:
            output_text += " " + sampled_word

        # Increment the token-counter.
        count_tokens += 1

    prediction_caption = output_text.split()
    # calculate matrics
    # Calculate BLEU-1 score
    bleu_1_train.append(sentence_bleu(caption_val_split, prediction_caption, weights=(1, 0, 0, 0), smoothing_function=chencherry.method1))
    # Calculate BLEU-2 score
    bleu_2_train.append(sentence_bleu(caption_val_split, prediction_caption, weights=(0.5, 0.5, 0, 0), smoothing_function=chencherry.method1))
    # Calculate BLEU-3 score
    bleu_3_train.append(sentence_bleu(caption_val_split, prediction_caption, weights=(1/3, 1/3, 1/3, 0), smoothing_function=chencherry.method1))
    # Calculate BLEU-4 score
    bleu_4_train.append(sentence_bleu(caption_val_split, prediction_caption, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=chencherry.method1))
    # calculate Meteor
    meteor_train.append(meteor_score(caption_val_split, prediction_caption))
    # rouge score
    for el in caption_val_raw:
        rouge_scores = scorer_rouge.score(el, ' '.join(prediction_caption))
        rouge_1_train.append(rouge_scores['rouge1'].fmeasure)
        rouge_2_train.append(rouge_scores['rouge2'].fmeasure)
        rouge_L_train.append(rouge_scores['rougeL'].fmeasure)

    counter += 1

print()
print("#"*20)
print("\tMetrics train set")
print("BLEU-1: ", np.round(np.mean(bleu_1_train),2), '+-', np.round(np.std(bleu_1_train),2))
print("BLEU-2: ", np.round(np.mean(bleu_2_train),2), '+-', np.round(np.std(bleu_2_train),2))
print("BLEU-3: ", np.round(np.mean(bleu_3_train),2), '+-', np.round(np.std(bleu_3_train),2))
print("BLEU-4: ", np.round(np.mean(bleu_4_train),2), '+-', np.round(np.std(bleu_4_train),2))
print("METEOR: ", np.round(np.mean(meteor_train),2), '+-', np.round(np.std(meteor_train),2))
print("ROUGE-1: ", np.round(np.mean(rouge_1_train),2), '+-', np.round(np.std(rouge_1_train),2))
print("ROUGE-2: ", np.round(np.mean(rouge_2_train),2), '+-', np.round(np.std(rouge_2_train),2))
print("ROUGE-L: ", np.round(np.mean(rouge_L_train),2), '+-', np.round(np.std(rouge_L_train),2))

#%%
# metrics calculation for validation set
cider_val = []
bleu_1_val = []; bleu_2_val = []; bleu_3_val = []; bleu_4_val = []
meteor_val = []
rouge_1_val = []; rouge_2_val = []; rouge_L_val = []
scorer_rouge = rouge_scorer.RougeScorer(['rouge1','rouge2', 'rougeL'], use_stemmer=True)
chencherry = SmoothingFunction()

image_dataset_val_numpy = np.array(image_dataset_val)
captions_dataset_val_numpy = np.array(captions_dataset_val)
images = set(image_dataset_val)

counter = 0
for element in images:
    utiles.print_progress(counter, len(images))

    index = np.where(image_dataset_val_numpy == element)
    # data
    image_val = np.load(element)
    reference_captions = captions_dataset_val_numpy[index]
    caption_val_raw = []
    caption_val_split = []
    for el in reference_captions: 
        cap = tokenizer.tokens_to_string(el[1:-1])
        caption_val_raw.append(cap)
        caption_val_split.append(cap.split())
    
    # make prediction
    shape = (1, CAPTION_LENGTH)
    decoder_input_data = np.zeros(shape=shape, dtype=np.int64)

    # The first input-token is the special start-token for 'ssss '.
    token_int = tokenizer.word_index[MARK_START.strip()]

    # Initialize an empty output-text.
    output_text = ''

    # Initialize the number of tokens we have processed.
    count_tokens = 0

    while token_int != tokenizer.word_index[MARK_END.strip()] and count_tokens < CAPTION_LENGTH:
        # Update the input-sequence to the decoder
        # with the last token that was sampled.
        # In the first iteration this will set the
        # first element to the start-token.
        decoder_input_data[0, count_tokens] = token_int

        # Wrap the input-data in a dict for clarity and safety,
        # so we are sure we input the data in the right order.
        x_data = {
            'transfer_values_input': image_val,
            'decoder_input': decoder_input_data}
        
        # Input this data to the decoder and get the predicted output.
        decoder_output = decoder_model.predict(x_data, verbose=0)

        # Get the last predicted token as a one-hot encoded array.
        # Note that this is not limited by softmax, but we just
        # need the index of the largest element so it doesn't matter.
        token_onehot = decoder_output[0, count_tokens, :]

        # Convert to an integer-token.
        token_int = np.argmax(token_onehot)

        # Lookup the word corresponding to this integer-token.
        sampled_word = tokenizer.token_to_word(token_int)

        # Append the word to the output-text.
        if sampled_word != token_end:
            output_text += " " + sampled_word

        # Increment the token-counter.
        count_tokens += 1

    prediction_caption = output_text.split()

    # calculate matrics
    # Calculate BLEU-1 score
    bleu_1_val.append(sentence_bleu(caption_val_split, prediction_caption, weights=(1, 0, 0, 0), smoothing_function=chencherry.method1))
    # Calculate BLEU-2 score
    bleu_2_val.append(sentence_bleu(caption_val_split, prediction_caption, weights=(0.5, 0.5, 0, 0), smoothing_function=chencherry.method1))
    # Calculate BLEU-3 score
    bleu_3_val.append(sentence_bleu(caption_val_split, prediction_caption, weights=(1/3, 1/3, 1/3, 0), smoothing_function=chencherry.method1))
    # Calculate BLEU-4 score
    bleu_4_val.append(sentence_bleu(caption_val_split, prediction_caption, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=chencherry.method1))
    # calculate Meteor
    meteor_val.append(meteor_score(caption_val_split, prediction_caption))
    # rouge score
    for el in caption_val_raw:
        rouge_scores = scorer_rouge.score(el, ' '.join(prediction_caption))
        rouge_1_val.append(rouge_scores['rouge1'].fmeasure)
        rouge_2_val.append(rouge_scores['rouge2'].fmeasure)
        rouge_L_val.append(rouge_scores['rougeL'].fmeasure)

    counter +=1

print()
print("#"*20)
print("\tMetrics val set")
print("BLEU-1: ", np.round(np.mean(bleu_1_val),2), '+-', np.round(np.std(bleu_1_val),2))
print("BLEU-2: ", np.round(np.mean(bleu_2_val),2), '+-', np.round(np.std(bleu_2_val),2))
print("BLEU-3: ", np.round(np.mean(bleu_3_val),2), '+-', np.round(np.std(bleu_3_val),2))
print("BLEU-4: ", np.round(np.mean(bleu_4_val),2), '+-', np.round(np.std(bleu_4_val),2))
print("METEOR: ", np.round(np.mean(meteor_val),2), '+-', np.round(np.std(meteor_val),2))
print("ROUGE-1: ", np.round(np.mean(rouge_1_val),2), '+-', np.round(np.std(rouge_1_val),2))
print("ROUGE-2: ", np.round(np.mean(rouge_2_val),2), '+-', np.round(np.std(rouge_2_val),2))
print("ROUGE-L: ", np.round(np.mean(rouge_L_val),2), '+-', np.round(np.std(rouge_L_val),2))

# %%
