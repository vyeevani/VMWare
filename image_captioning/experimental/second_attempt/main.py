import os
import tensorflow as tf
import numpy as tf

model_path = './models/tensorflow'
vgg_path = './data/vgg16-20160129.tfmodel'
image_paths = ['./data/images/img1', './data/image/img2']

dim_embeddim_emb  = 256
dim_hidden = 256
dim_in = 4096
batch_size = 1
learning_rate = 0.001
momentum = 0.9
n_epochs = 25

if not os.path.exists('data/ixtoword.npy'):
    print ('You must run 1. O\'reilly Training.ipynb first.')
else:
    tf.reset_default_graph()
    with open(vgg_path,'rb') as f:
        fileContent = f.read()
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(fileContent)

    images = tf.placeholder("float32", [1, 224, 224, 3])
    tf.import_graph_def(graph_def, input_map={"images":images})

    ixtoword = np.load('data/ixtoword.npy').tolist()
    n_words = len(ixtoword)
    maxlen=15
    graph = tf.get_default_graph()
    sess = tf.InteractiveSession(graph=graph)
    caption_generator = Caption_Generator(dim_in, dim_hidden, dim_embed, batch_size, maxlen+2, n_words)
    graph = tf.get_default_graph()
    image, generated_words = caption_generator.build_generator(maxlen=maxlen)
    for image_path in image_paths:
        test(sess,image,generated_words,ixtoword, image_path)
