import tensorflow as tf
import tensorflow.contrib.keras as keras


video = keras.layers.Input(shape=(None, 150, 150, 3))
cnn = keras.applications.InceptionV3(weights='imagenet', include_top=False, pool='avg')

cnn.tranable = False
encoded_trames = keras.layers.TimeDistributed(cnn)(video)
encoded_vid = layers.LSTM(256)(encoded_frames)

question = keras.layers.Input(shape=(100), dtype='int32')
x = keras.layers.Embedding(10000, 256, mask_zero=True)(question)
encoded_q = keras.layers.LSTM(128)(x)

x = keras.layers.concat([encoded_vid, encoded_q])
x = tf.keras.layers.Dense(128, activation=tf.nn.relu)(x)
outputs = keras.layers.Dense(1000)(x)

model = keras.models.Model([video, question], outputs)
model.complile(optimizer = tf.AdamOptimizer(), loss = tf.softmax_crossentropy_with_logits)
