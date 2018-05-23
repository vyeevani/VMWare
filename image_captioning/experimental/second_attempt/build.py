from utils.train import train
from utils.get_data import get_data

model_path = './models/tensorflow'
model_path_transfer = './models/tf_final'
feature_path = './data/feats.npy'
annotation_path = './data/results_20130124.token'
print(get_data)
feats, captions = get_data(annotation_path, feature_path)
print(feats.shape)
print(captions.shape)
print(captions[0])

try:
    train(.001,False,False) #train from scratch
    #train(.001,True,True)    #continue training from pretrained weights @epoch500
    #train(.001,True,False)  #train from previously saved weights
except KeyboardInterrupt:
    print('Exiting Training')
