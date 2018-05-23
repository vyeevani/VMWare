import numpy as np
import pandas as pd
import tensorflow as tf

def get_data(annotation_path, feature_path):
    annotations = pd.read_table(annotation_path, sep='\t', header=None, names=['image', 'caption'])
    return np.load(feature_path, 'r'), annotations['caption'].values
