import tensorflow as tf
import numpy as np
from image import read_image

model_path = "./models/tensorflow"

def test(sess,image,generated_words,ixtoword,test_image_path, graph, images): # Naive greedy search
    feat = read_image(test_image_path)
    fc7 = sess.run(graph.get_tensor_by_name("import/Relu_1:0"), feed_dict={images:feat})

    saver = tf.train.Saver()
    sanity_check=False
    # sanity_check=True
    if not sanity_check:
        saved_path=tf.train.latest_checkpoint(model_path)
        saver.restore(sess, saved_path)
    else:
        tf.global_variables_initializer().run()

    generated_word_index= sess.run(generated_words, feed_dict={image:fc7})
    generated_word_index = np.hstack(generated_word_index)
    generated_words = [ixtoword[x] for x in generated_word_index]
    punctuation = np.argmax(np.array(generated_words) == '.')+1

    generated_words = generated_words[:punctuation]
    generated_sentence = ' '.join(generated_words)
    return(generated_sentence)
