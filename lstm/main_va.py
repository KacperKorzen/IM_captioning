#%%
from pycocoevalcap.cider.cider import Cider
from rouge_score import rouge_scorer

from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from nltk.translate.bleu_score import SmoothingFunction

from tensorflow.keras.preprocessing.sequence import pad_sequences

import time
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

from PIL import Image
import pickle

import utiles 
import utiles_va
from sklearn.model_selection import train_test_split

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

'''
def standardize(inputs):
  inputs = tf.strings.lower(inputs)
  return tf.strings.regex_replace(inputs,
                                  r"!\"#$%&\(\)\*\+.,-/:;=?@\[\\\]^_`{|}~", "")
import tensorflow as tf
tokenizer = tf.keras.layers.TextVectorization(
    max_tokens=5000,
    standardize=standardize,
    output_sequence_length=33)


tokenizer.adapt(flat)
'''

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
# %% ekstrakcja cech 
if COCO:
    images_features, features_shape, attention_features_shape = utiles_va.extract_features_VGG16(PATH_COCO_IMAGES+'/train2017')
else:
    #images_features, features_shape, attention_features_shape = utiles_va.extract_features_VGG16(PATH_FLIKER_8K_IMAGES)
    images_features, _, _ = utiles_va.extract_features_xcepction(PATH_FLIKER_8K_IMAGES)

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

path_to_save = PATH_FLIKER_8K_IMAGES[:-6] + 'xception_'
with open(path_to_save + 'train_caption.pkl', 'wb') as fp:
    pickle.dump(train_captions, fp)
with open(path_to_save + 'train_images.pkl', 'wb') as fp:
    pickle.dump(train_images, fp)

with open(path_to_save + 'test_caption.pkl', 'wb') as fp:
    pickle.dump(test_captions, fp)
with open(path_to_save + 'test_images.pkl', 'wb') as fp:
    pickle.dump(test_images, fp)


#%%
#########################################
#########################################
#########################################
#########################################

# set path to load elements
path_to_save = PATH_FLIKER_8K_IMAGES[:-6] + 'vgg_'#'xception_'
with open(path_to_save + 'train_caption.pkl', 'rb') as fp:
    train_captions = pickle.load(fp)
with open(path_to_save + 'train_images.pkl', 'rb') as fp:
    train_images = pickle.load(fp)
with open(path_to_save + 'test_caption.pkl', 'rb') as fp:
    test_captions = pickle.load(fp)
with open(path_to_save + 'test_images.pkl', 'rb') as fp:
    test_images = pickle.load(fp)

# load
BUFFER_SIZE = 1000
embedding_dim = 256
units = 256#256#512
# Shape of the vector extracted from InceptionV3 is (64, 2048)
# These two variables represent that vector shape
shapes = np.load(train_images[list(train_images.keys())[0]]).shape
features_shape = shapes[2] #features_shape #train_images[train_keys[0]].shape[2]
attention_features_shape = shapes[1]#attention_features_shape#train_images[train_keys[0]].shape[1]

CAPTION_LENGTH = 0
for key, value in train_captions.items():
    for inner_list in value:
        if len(inner_list) > CAPTION_LENGTH:
            CAPTION_LENGTH = len(inner_list)
#%%
encoder = utiles_va.CNN_Encoder(embedding_dim)
decoder = utiles_va.RNN_Decoder(embedding_dim, units, NUM_WORDS)

optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')


def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask

  return tf.reduce_mean(loss_)

#%%
'''
checkpoint_path = "./checkpoints/train/Va"
ckpt = tf.train.Checkpoint(encoder=encoder,
                           decoder=decoder,
                           optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
'''
start_epoch = 0
'''
if ckpt_manager.latest_checkpoint:
    start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
    # restoring the latest checkpoint in checkpoint_path
    ckpt.restore(ckpt_manager.latest_checkpoint)
'''
# adding this in a separate cell because if you run the training cell
# many times, the loss_plot array will be reset
loss_plot = []
val_loss_plot = []
#%%
@tf.function
def train_step(img_tensor, target):
    loss = 0

    # initializing the hidden state for each batch
    # because the captions are not related from image to image
    hidden_state = decoder.reset_state(batch_size=target.shape[0])

    hidden = [hidden_state, hidden_state]


    dec_input = tf.expand_dims([token_start] * target.shape[0], 1)

    with tf.GradientTape() as tape:
        features = encoder(img_tensor)

        for i in range(1, target.shape[1]):
            # passing the features through the decoder
            predictions, hidden, _ = decoder(dec_input, features, hidden)

            loss += loss_function(target[:, i], predictions)

            # using teacher forcing
            dec_input = tf.expand_dims(target[:, i], 1)

    total_loss = (loss / int(target.shape[1]))

    trainable_variables = encoder.trainable_variables + decoder.trainable_variables

    gradients = tape.gradient(loss, trainable_variables)

    optimizer.apply_gradients(zip(gradients, trainable_variables))

    return loss, total_loss

@tf.function
def validation_step(img_tensor, target):
    loss = 0

    # initializing the hidden state for each batch
    # because the captions are not related from image to image
    hidden_state = decoder.reset_state(batch_size=target.shape[0])
    hidden = [hidden_state, hidden_state]

    dec_input = tf.expand_dims([token_start] * target.shape[0], 1)

    features = encoder(img_tensor)

    for i in range(1, target.shape[1]):
        # passing the features through the decoder
        predictions, hidden, _ = decoder(dec_input, features, hidden)

        loss += loss_function(target[:, i], predictions)

        # using teacher forcing
        dec_input = tf.expand_dims(target[:, i], 1)

    total_loss = (loss / int(target.shape[1]))

    return loss, total_loss

#%%
# training
captions_dataset_train = []
image_dataset_train = []

def map_func(img_name, cap):
    img_tensor = np.load(img_name)
    img_tensor = np.reshape(img_tensor, [img_tensor.shape[1], img_tensor.shape[2]])
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

#%%
for epoch in range(start_epoch, EPOCHS):
    start = time.time()
    total_loss = 0
    total_loss_vall = 0

    for (batch, (img_tensor, target)) in enumerate(dataset):
        batch_loss, t_loss = train_step(img_tensor.numpy(), target.numpy())
        total_loss += t_loss

        if batch % 100 == 0:
            average_batch_loss = batch_loss.numpy()/int(target.shape[1])
            print(f'Epoch {epoch+1} Batch {batch} Loss {average_batch_loss:.4f}')
    # storing the epoch end loss value to plot later
    loss_plot.append(total_loss / num_steps_train)
    
    for (batch_val, (img_tensor, target)) in enumerate(dataset):
        val_loss_total, val_loss = validation_step(img_tensor, target)
        total_loss_vall += val_loss
    val_loss_plot.append(total_loss_vall / num_steps_val)


    if epoch % 5 == 0:
        #ckpt_manager.save()
        pass

    print(f'Epoch {epoch+1} Loss train {total_loss/num_steps_train:.6f} Loss val {val_loss_total/num_steps_val:.6f}')
    print(f'Time taken for 1 epoch {time.time()-start:.2f} sec\n')
   

#%% sample caption

def evaluate(path_to_photo):
    attention_plot = np.zeros((CAPTION_LENGTH, attention_features_shape))
    
    hidden_state = decoder.reset_state(batch_size=1)
    hidden = [hidden_state, hidden_state]
    '''
    model_image = VGG16(include_top = True)
    
    # pop last fully connect layer
    model_image = Model(inputs = model_image.input,
                outputs = model_image.layers[-5].output)

    file_path = path_to_photo
    image = load_img(file_path, target_size = (model_image.input.shape[1],model_image.input.shape[2]))
    # convert the image pixels to a numpy array
    image = img_to_array(image, dtype = 'float16')
    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # prepare the image for the VGG model
    image = preprocess_input(image)
    
    feature = model_image(image)

    img_tensor_val = tf.reshape(feature,
                        (feature.shape[0], -1, feature.shape[3]))
    features = encoder(img_tensor_val)
    '''
    img_tensor = np.load(path_to_photo)
    features = encoder(img_tensor)
    dec_input = tf.expand_dims([tokenizer.word_index[MARK_END.strip()]], 0)
    result = []

    for i in range(CAPTION_LENGTH):
        predictions, hidden, attention_weights = decoder(dec_input,
                                                         features,
                                                         hidden)

        attention_plot[i] = tf.reshape(attention_weights, (-1, )).numpy()

        predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
        predicted_word = tf.compat.as_text(tokenizer.index_to_word[predicted_id])
        result.append(predicted_word)

        if predicted_word == MARK_END:
            return result, attention_plot

        dec_input = tf.expand_dims([predicted_id], 0)

    attention_plot = attention_plot[:len(result), :]
    return result, attention_plot

#%%
result, attention_plot = evaluate(PATH_FLIKER_8K_IMAGES[:-6]+'features_vgg_va'+'/190638179_be9da86589.npy')

temp_image = np.array(Image.open(PATH_FLIKER_8K_IMAGES+'/190638179_be9da86589.jpg'))
fig = plt.figure(figsize=(10, 10))

len_result = len(result)
for i in range(len_result):
    temp_att = np.resize(attention_plot[i], (int(np.sqrt(attention_features_shape)), int(np.sqrt(attention_features_shape))))
    grid_size = max(int(np.ceil(len_result/2)), 2)
    ax = fig.add_subplot(grid_size, grid_size, i+1)
    ax.set_title(result[i])
    img = ax.imshow(temp_image)
    ax.imshow(temp_att, cmap='gray', alpha=0.6, extent=img.get_extent())

plt.tight_layout()
plt.show()

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
    hidden_state = decoder.reset_state(batch_size=1)
    hidden = [hidden_state, hidden_state]
    features = encoder(image_val)
    dec_input = tf.expand_dims([tokenizer.word_index[MARK_END.strip()]], 0)
    prediction_caption = []

    for i in range(CAPTION_LENGTH):
        predictions, hidden, attention_weights = decoder(dec_input,
                                                         features,
                                                         hidden)

        predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
        predicted_word = tf.compat.as_text(tokenizer.index_to_word[predicted_id])
        prediction_caption.append(predicted_word)

        if predicted_word == MARK_END:
            prediction_caption = prediction_caption[:-1]
            break
            
        dec_input = tf.expand_dims([predicted_id], 0)

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
#print("CIDER: ", np.round(np.mean(cider_train),2), '+-', np.round(np.std(cider_train),2))

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
    hidden_state = decoder.reset_state(batch_size=1)
    hidden = [hidden_state, hidden_state]
    features = encoder(image_val)
    dec_input = tf.expand_dims([tokenizer.word_index[MARK_END.strip()]], 0)
    prediction_caption = []

    for i in range(CAPTION_LENGTH):
        predictions, hidden, attention_weights = decoder(dec_input,
                                                         features,
                                                         hidden)

        predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
        predicted_word = tf.compat.as_text(tokenizer.index_to_word[predicted_id])
        prediction_caption.append(predicted_word)

        if predicted_word == MARK_END:
            prediction_caption = prediction_caption[:-1]
            break
            
        dec_input = tf.expand_dims([predicted_id], 0)

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
# create figure and axis objects with subplots()
fig,ax = plt.subplots()
# make a plot
ax.plot(loss_plot, color="red", marker="o")
# set x-axis label
ax.set_xlabel("Epochs", fontsize = 14)
# set y-axis label
ax.set_ylabel("Train Loss", color="red", fontsize=14)

# twin object for two different y-axis on the sample plot
ax2=ax.twinx()
# make a plot with different y-axis using second axis object
ax2.plot(val_loss_plot, color="blue", marker="x")
ax2.set_ylabel("Validation Loss", color="blue", fontsize=14)
plt.show()

#%%
print('end')