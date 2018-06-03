import tensorflow as tf
import numpy as np
import os
from keras.preprocessing import sequence
from caption_generator import Caption_Generator
from build_vocab import preProBuildWordVocab
from get_data import get_data

def train(learning_rate=0.001, continue_training=False, transfer=True, server=None, is_chief=False, device_function=None):
    with tf.device(device_function):
        annotation_path = './data/results_20130124.token'
        feature_path = './data/feats.npy'
        model_path = './models/tensorflow'
        model_path_transfer = './data/feats.npy'
        ### Parameters ###
        dim_embed = 256
        dim_hidden = 256
        dim_in = 4096
        batch_size = 128
        momentum = 0.9
        n_epochs = 150

        tf.reset_default_graph()
        global_step = tf.train.get_or_create_global_step()
        hooks = [tf.train.StopAtStepHook(n_epochs)]

        feats, captions = get_data(annotation_path, feature_path)
        wordtoix, ixtoword, init_b = preProBuildWordVocab(captions)

        np.save('data/ixtoword', ixtoword)

        index = (np.arange(len(feats)).astype(int))
        np.random.shuffle(index)
        n_words = len(wordtoix)
        maxlen = np.max( [x for x in map(lambda x: len(x.split(' ')), captions) ] )
        caption_generator = Caption_Generator(dim_in, dim_hidden, dim_embed, batch_size, maxlen+2, n_words, device_function ,init_b)

        loss, image, sentence, mask = caption_generator.build_model()

        saver = tf.train.Saver(max_to_keep=100)
        global_step=tf.Variable(0,trainable=False)
        learning_rate = tf.train.exponential_decay(learning_rate, global_step,
                                           int(len(index)/batch_size), 0.95)
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)


        with tf.train.MonitoredTrainingSession(server.target, is_chief=is_chief, checkpoint_dir="models/tmp/train_logs",
                                       hooks=hooks) as sess:
            #sess.run(tf.global_variables_initializer())

            if continue_training:
                if not transfer:
                    saver.restore(sess,tf.train.latest_checkpoint(model_path))
                else:
                    saver.restore(sess,tf.train.latest_checkpoint(model_path_transfer))
            losses=[]
            #for epoch in range(n_epochs):
            epoch = 0
            while not sess.should_stop():
                for start, end in zip( range(0, len(index), batch_size), range(batch_size, len(index), batch_size)):

                    current_feats = feats[index[start:end]]
                    current_captions = captions[index[start:end]]
                    current_caption_ind = [x for x in map(lambda cap: [wordtoix[word] for word in cap.lower().split(' ')[:-1] if word in wordtoix], current_captions)]

                    current_caption_matrix = sequence.pad_sequences(current_caption_ind, padding='post', maxlen=maxlen+1)
                    current_caption_matrix = np.hstack( [np.full( (len(current_caption_matrix),1), 0), current_caption_matrix] )

                    current_mask_matrix = np.zeros((current_caption_matrix.shape[0], current_caption_matrix.shape[1]))
                    nonzeros = np.array([x for x in map(lambda x: (x != 0).sum()+2, current_caption_matrix )])

                    for ind, row in enumerate(current_mask_matrix):
                        row[:nonzeros[ind]] = 1

                    _, loss_value = sess.run([train_op, loss], feed_dict={
                        image: current_feats.astype(np.float32),
                        sentence : current_caption_matrix.astype(np.int32),
                        mask : current_mask_matrix.astype(np.float32)
                        })

                    print("Current Cost: ", loss_value, "\t Epoch {}/{}".format(epoch, n_epochs), "\t Iter {}/{}".format(start,len(feats)))
                print("Saving the model from epoch: ", epoch)
                epoch += 1
                saver.save(sess, os.path.join(model_path, 'model'), global_step=epoch)
