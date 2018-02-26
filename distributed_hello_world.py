# Machine Learning library imports
# THIS FILE WILL NOT RUN WRITTEN AS A DEMONSTRATION
import tensorflow as tf
from tensorflow.contrib.keras.api.keras.layers import LSTM, Embedding, TimeDistributed, RepeatVector, LSTM, concatenate , Input, Reshape, Dense
from tensorflow.contrib.keras.api.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.contrib.keras.api import keras as keras
from tensorflow.contrib.keras.api.keras.preprocessing.image import array_to_img, img_to_array, load_img
from tensorflow.contrib.keras.api.keras.models import Model

# Import numpy
import numpy as np

# Declare inputs
max_caption_len = 3
vocab_size = 3

html_input = np.array(
            [[[0., 0., 0.],
             [0., 0., 0.],
             [1., 0., 0.]],
             [[0., 0., 0.],
             [1., 0., 0.],
             [0., 1., 0.]]])

next_words = np.array(
            [[0., 1., 0.], # <HTML>Hello World!</HTML>
             [0., 0., 1.]]) # end

filename = "screenshot.jpg"

# Load the screenshot
images = []
for i in range(2):
    images.append(img_to_array(load_img('screenshot.jpg', target_size=(224, 224))))
images = np.array(images, dtype=float)

# Preprocess input for the VGG16 model
images = preprocess_input(images)

# get features from VGG16 model
VGG = VGG16(weights='imagenet', include_top=True)
features = VGG.predict(images)

# Learn from features
vgg_feature = Input(shape=(1000,))
vgg_feature_dense = Dense(5)(vgg_feature)
vgg_feature_repeat = RepeatVector(max_caption_len)(vgg_feature_dense)

# Learn from language
language_input = Input(shape=(vocab_size, vocab_size))
language_model = keras.layers.LSTM(5, return_sequences=True)(language_input)

# Combine the features
decoder = concatenate([vgg_feature_repeat, language_model], axis=1)

# Learn from combination
decoder = keras.layers.LSTM(5, return_sequences=False)(decoder)

# Predict which word comes next
decoder_output = Dense(vocab_size, activation='softmax')(decoder)

# Train the model on seperate GPUs with seperate inputs
# Must add different training inputs in later models.
with tf.device('/gpu:0'):
    model1 = Model(inputs=[vgg_feature, language_input], outputs=decoder_output)
    model1.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    model1.fit([features, html_input], next_words, batch_size=2, shuffle=False, epochs=1000)

with tf.device('/gpu:1'):
    model2 = Model(inputs=[vgg_feature, language_input], outputs=decoder_output)
    model2.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    model2.fit([features, html_input], next_words, batch_size=2, shuffle=False, epochs=1000)

# Average all outputs from all distributed models
def model_predict(input):
    return (model1.predict(input) + model2.predict(input)) * 0.5

start_token = [1., 0., 0.] # start
sentence = np.zeros((1, 3, 3)) # [[0,0,0], [0,0,0], [0,0,0]]
sentence[0][2] = start_token # place start in empty sentence

# Making the first prediction with the start token
second_word = model_predict([np.array([features[1]]), sentence])

# Put the second word in the sentence and make the final prediction
sentence[0][1] = start_token
sentence[0][2] = np.round(second_word)
third_word = model_predict([np.array([features[1]]), sentence])

sentence[0][0] = start_token
sentence[0][1] = np.round(second_word)
sentence[0][2] = np.round(third_word)

# Transform our one-hot predictions into the final tokens
vocabulary = ["start", "<HTML><center><H1>Hello World!</H1><center></HTML>", "end"]
html = ""
for i in sentence[0]:
    html += vocabulary[np.argmax(i)] + ' '

print(html)